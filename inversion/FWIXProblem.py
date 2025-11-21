import pandas as pd
import numpy as np
import dask
import pyVector as Vector
import pyOperator as Operator
from pyProblem import Problem  # Assuming base class is in pyProblem.py
from pyParquetVector import pyParquetVector  # Assuming this is your Dask-compatible vector
from .helper import _obj_grad, _obj
from fwix.operator import Derivative
import SepVector
import Hypercube
import genericIO

import pyOperator as Op
import pyVector as Vec
import pyLinearSolver as LinearSolver
import pyStopper as Stopper
import pyProblem as Prblm
import pyproximal as pp
from pyProxOperator import ProxOperatorExplicit, ProxDstack

class FWIXProblem(Prblm.Problem):
    def __init__(self, 
                 par: dict,
                 prop_par: genericIO.pythonParams, 
                 raw_model: Vector.vector, 
                 raw_density: Vector.vector,
                 data: pyParquetVector,
                 wavelet: SepVector.Vector,
                 resource_constraints: dict = None):
        
        super(FWIXProblem, self).__init__()
        
        self.data = data
        self.prop_par = prop_par
        self.wavelet = wavelet
        self.resource_constraints = resource_constraints or {'GPU': 1}
        # Disable Dask fusion optimization to allow resouce annotations to work properly
        dask.config.set({"optimization.fuse.active": False})

        # --- 1. Build Physical Model Space (High Res) ---
        self.phys_model = Vec.superVector(raw_model, raw_density)
        self.phys_grad = self.phys_model.clone().zero()

        # --- 2. Build Preconditioner (Lanczos/Spline) ---
        self.precond_op = None
        
        if 'pre' in par:
            print("Building Preconditioner...")
            ops_pre = []
            vecs_pre = []
            
            # Iterate over [Vel, Den]
            for i, mod in enumerate(self.phys_model.vecs):
                ax = mod.getHyper().axes
                ns_pre = par['pre']['ns'][i]
                ds_pre = [(ax[j].n-1)*ax[j].d / (ns_pre[j] - 1) for j in range(len(ns_pre))]
                
                # Create Coarse Vector
                mod_pre = SepVector.getSepVector(
                    Hypercube.hypercube(ns=ns_pre, ds=ds_pre, os=[a.o for a in ax]), 
                    storage='dataComplex'
                )
                mod_pre.zero()
                vecs_pre.append(mod_pre)

                # Build Interpolator
                interp = Operator.Spline3D(mod_pre, mod, type="CR-spline")
                ops_pre.append(interp)
            
            # Define Optimization Variable (Low Res p)
            self.model = Vec.superVector(vecs_pre[0], vecs_pre[1])
            
            # Preconditioner is Diagonal Stack (Op1 on Vec1, Op2 on Vec2)
            self.precond_op = Op.Dstack(self.model, self.phys_model, ops_pre)
            
            # --- 3. Initialize 'p' via Linear Inversion ---
            print("Initializing Preconditioned Model via Linear CGLS...")
            # We assume the physical start model is 'correct', we need the 'p' that generates it
            # minimize || S * p - m_start ||^2
            LinStop  = Stopper.BasicStopper(niter=par['pre']['niter'])
            CGsolver = LinearSolver.LCGsolver(LinStop)
            # Note: We use standard ProblemL2Linear because models are small (fit in memory)
            InitProb = Prblm.ProblemL2Linear(self.model, self.phys_model, self.precond_op)
            CGsolver.setDefaults(save_obj=False, save_res=False, save_grad=False, save_model=False)
            CGsolver.run(InitProb, verbose=True)
            
        else:
            # No Preconditioning
            self.model = self.phys_model.clone()
            self.precond_op = None

        # --- 4. Setup Proximal Operator (Bounds on Slowness) ---
        self.proxOp = None
        if ('pre' in par) and ("vmin" in par) and ("vmax" in par):
            print("Computing Proximal Bounds in Preconditioned Space...")
            slim = []
            for lim in ["vmax", "vmin"]:
                # Create a physical vector of 1/v^2
                s_phys = self.phys_model.vecs[0].clone()
                s_phys.set(1.0/par[lim]**2)
                
                # Solve for the equivalent bound in 'p' space
                # minimize || S_vel * p_vel - s_phys_lim ||
                sub = self.model.vecs[0].clone().zero()
                BoundProb = Prblm.ProblemL2Linear(sub, s_phys, self.precond_op.ops[0])
                CGsolver.run(BoundProb, verbose=False)
                slim.append(sub) # slim[0] is max_bound (from vmax), slim[1] is min_bound
            
            # Create Box Constraint on Velocity (Optimization Variable)
            vel_prox = ProxOperatorExplicit(pp.Box(lower=slim[0][:], upper=slim[1][:]))
            self.proxOp = ProxDstack([vel_prox, None])

        # --- 5. Setup Regularization ---
        # Regularization acts on PHYSICAL model, but gradient propagated to PRECONDITIONED
        self.reg_op = None
        self.epsilon = par["reg"]["epsilon"] if "reg" in par else 0.0
        
        if self.epsilon > 0:
            dt = self.wavelet.getHyper().getAxis(1).d
            tmax = (self.wavelet.getHyper().getAxis(1).n - 1)*dt + self.wavelet.getHyper().getAxis(1).o
            
            # Derivative on Physical Velocity
            m_vel = self.phys_model.vecs[0]
            self.reg_op = Operator.Derivative(m_vel, m_vel, which=1, 
                                          order=4, mode=par["reg"]['mode'])
            
            # Store it
            self.reg_vec_temp = m_vel.clone()

        # Initialize Gradients
        self.grad = self.model.clone().zero()
        self.dmodel = self.model.clone().zero() # Temp vector
        
        self.setDefaults()

    def objgradf(self, model):
        """
        Chain Rule Application:
        p -> [Precond] -> m_phys -> [FWI] -> J_data
                                 -> [Reg] -> J_reg
        """
        
        # --- 1. Map to Physical Domain ---
        if self.precond_op:
            self.precond_op.forward(False, model, self.phys_model)
        else:
            self.phys_model.copy(model)

        # --- 2. FWI Data Misfit (Out-of-Core) ---

        meta_df = pd.DataFrame({'norm_sq': pd.Series(dtype='float64'),
                                'grad': pd.Series(dtype='object')})
        
        with dask.annotate(resources=self.resource_constraints):
            res_df = self.data.df[self.data.key].map_partitions(
                _obj_grad,
                self.phys_model, 
                self.wavelet,
                self.prop_par,
                meta=meta_df
            )

        summed_df = res_df.sum().compute()
        
        self.obj = 0.5 * summed_df["norm_sq"]
        # phys_grad = \nabla_{m_phys} J_data
        self.phys_grad.copy(summed_df["grad"]) 

        # --- 3. Regularization (Physical Domain) ---
        if self.reg_op:
            # J_reg = 0.5 * eps^2 * || D(m_vel) ||^2
            m_vel = self.phys_model.vecs[0]
            
            self.reg_op.forward(False, m_vel, self.reg_vec_temp)
            self.obj += 0.5 * (self.epsilon**2) * self.reg_vec_temp.norm()**2
            
            # Grad_reg = eps^2 * D' D m_vel
            self.reg_vec_temp.scale(self.epsilon**2)
            
            g_vel_reg = m_vel.clone().zero()
            self.reg_op.adjoint(True, g_vel_reg, self.reg_vec_temp)
            
            # Add regularization gradient to physical gradient
            self.phys_grad.scaleAdd(g_vel_reg, 1.0, 1.0)
            del g_vel_reg

        # --- 4. Backpropagate to Optimization Domain (p) ---
        if self.precond_op:
            # \nabla_p J = S^T * (\nabla_{m_phys} J_data + \nabla_{m_phys} J_reg)
            self.precond_op.adjoint(False, self.grad, self.phys_grad)
        else:
            self.grad.copy(self.phys_grad)

        self.obj_updated = True
        self.grad_updated = True
        self.fevals += 1
        self.gevals += 1

        return self.obj, self.grad
    
    def get_obj(self, model):
        """Overrides base class 'get_obj' to use our fused Dask method."""
        self.set_model(model)
        if not self.obj_updated:
            # Define the output structure for map_partitions
            meta_df = pd.DataFrame({'norm_sq': pd.Series(dtype='float64')})

            # Run the map-partitions (map) and sum (reduce)
            with dask.annotate(resources=self.resource_constraints):
                res_df = self.data.map_partitions(
                    _obj,
                    model, 
                    self.wavelet,
                    self.prop_par,
                    meta=meta_df
                )

            # compute() triggers the Dask computation and returns a pandas DataFrame
            summed_df = res_df.sum().compute()
            self.obj = 0.5 * summed_df["norm_sq"]
        return self.obj

    # --- Override essential base class methods ---

    def get_grad(self, model):
        """Overrides base class 'get_grad' to use our fused Dask method."""
        self.set_model(model)
        if not self.grad_updated:
            # This calls objgradf and sets self.obj and self.grad
            _, self.grad = self.get_obj_grad(model)
        return self.grad
        
    def get_obj_grad(self, model):
        """
        Accessor for objective function and gradient vector.
        This overrides the base method to call our specific objgradf.
        """
        self.set_model(model)
        if not self.obj_updated:
            # This is the only place the computation is triggered
            self.obj, self.grad = self.objgradf(model)
        return self.obj, self.grad

    # --- Ban methods that materialize the full residual ---

    def resf(self, model):
        """Residual vector is too large to compute and store."""
        raise NotImplementedError("DaskParquetProblem does not support materializing the full residual vector!")

    def dresf(self, model, dmodel):
        """dres is also too large to materialize."""
        raise NotImplementedError("DaskParquetProblem does not support dres computation!")

    def get_res(self, model):
        """Accessor for residual vector."""
        raise NotImplementedError("DaskParquetProblem does not support get_res!")