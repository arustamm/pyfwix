import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import gc
from typing import Tuple, Dict, Any

import SepVector
import Hypercube
import genericIO
import pyVector as Vec
import pyOperator as Op
import pyProblem as Prblm
import pyLinearSolver as LinearSolver
import pyStopper as Stopper
import pyproximal as pp
from pyProxOperator import ProxOperatorExplicit, ProxDstack
from fwix.workers import _obj_grad_worker, _obj_worker, _reduce_vector

from fwix import CudaOperator

class FWIXProblem(Prblm.Problem):
	def __init__(self, 
				start_model: Vec.vector, 
				start_density: Vec.vector,
				data_pipeline, # pysep3d pipeline object
				prop_par: Dict[str, Any],
				wavelet: pd.DataFrame,
				problem_par: dict,
				shots_per_gpu: int = 1,
				gpu_stream_batches: Tuple[int] = (1, 1),
				geometry_mapping: Dict[str, str] = {
					"sx": "sx",
					"sy": "sy",
					"sz": "sz",
					"id": "uniqueshots",
					"rx": "rx",
					"ry": "ry",
					"rz": "rz",
				},
				retry_tasks: int = 3,
			):
		
		super(FWIXProblem, self).__init__()
		
		# Data and Parameters
		# create the reading pipeline (i.e., read + float_to_complex conversion)
		self.data = data_pipeline.execute(return_pandas=False) # dask dataframe
		self.prop_par = prop_par
		self.problem_par = problem_par
		self.retry_tasks = retry_tasks

		if not wavelet[geometry_mapping['id']].is_monotonic_increasing:
			wavelet = wavelet.sort_values(geometry_mapping['id'])
		wavelet = wavelet.set_index(geometry_mapping['id'])
		self.wavelet = wavelet
		
		# Batching Config (Crucial for Memory)
		self.shots_per_gpu = shots_per_gpu
		self.gpu_stream_batches = gpu_stream_batches
		self.geometry_mapping = geometry_mapping

		# --- 1. Build Physical Model Space (High Res) ---
		self.phys_model = Vec.superVector(start_model, start_density)
		self.phys_grad = self.phys_model.clone().zero()
		#  get ginsu parameters for padding the model for each shot batch

		# --- 2. Build Preconditioner (Lanczos/Spline) ---
		self.precond_op = None
		
		if 'pre' in problem_par:
			print("Building Preconditioner...")
			ops_pre = []
			vecs_pre = []
			
			# Iterate over [Vel, Den]
			for i, mod in enumerate(self.phys_model.vecs):
				ax = mod.getHyper().axes
				ns_pre = problem_par['pre']['ns'][i]
				ds_pre = [(ax[j].n-1)*ax[j].d / (ns_pre[j] - 1) for j in range(len(ns_pre))]
				
				# Create Coarse Vector
				mod_pre = SepVector.getSepVector(
					Hypercube.hypercube(ns=ns_pre, ds=ds_pre, os=[a.o for a in ax]), 
					storage='dataComplex'
				)
				mod_pre.zero()
				vecs_pre.append(mod_pre)

				# Build Interpolator
				interp = CudaOperator.Spline4D(mod_pre, mod, type="CR-spline")
				ops_pre.append(interp)
			
			# Define Optimization Variable (Low Res p)
			self.model = Vec.superVector(vecs_pre[0], vecs_pre[1])
			self.precond_op = Op.Dstack(self.model, self.phys_model, ops_pre)
			
			# Initialize 'p' via Linear Inversion (CGLS)
			print("Initializing Preconditioned Model via Linear CGLS...")
			LinStop  = Stopper.BasicStopper(niter=problem_par['pre']['niter'])
			CGsolver = LinearSolver.LCGsolver(LinStop)
			InitProb = Prblm.ProblemL2Linear(self.model, self.phys_model, self.precond_op)
			CGsolver.setDefaults(save_obj=False, save_res=False, save_grad=False, save_model=False)
			CGsolver.run(InitProb, verbose=True)
			
		else:
			self.model = self.phys_model.clone()
			self.precond_op = None

		# --- 3. Setup Proximal Operator ---
		self.proxOp = None
		if ('pre' in problem_par) and ("vmin" in problem_par) and ("vmax" in problem_par):
			print("Computing Proximal Bounds...")
			slim = []
			for lim in ["vmax", "vmin"]:
				s_phys = self.phys_model.vecs[0].clone()
				s_phys.set(1.0/problem_par[lim]**2)
				sub = self.model.vecs[0].clone().zero()
				BoundProb = Prblm.ProblemL2Linear(sub, s_phys, self.precond_op.ops[0])
				CGsolver.run(BoundProb, verbose=False)
				slim.append(sub) 
			
			vel_prox = ProxOperatorExplicit(pp.Box(lower=slim[0][:], upper=slim[1][:]))
			self.proxOp = ProxDstack([vel_prox, None])

		# --- 4. Setup Regularization ---
		self.reg_op = None
		self.epsilon = problem_par.get("reg", {}).get("epsilon", 0.0)
		
		if self.epsilon > 0:
			self.reg_op = CudaOperator.Derivative(self.phys_model, self.phys_model, which=1, 
										  order=4, mode=problem_par["reg"]['mode'])
			self.reg_vec = self.phys_model.clone()

		# Initialize Gradients
		self.grad = self.model.clone().zero()
		self.grad_mask = problem_par.get("grad_mask", None)
		self.dmodel = self.model.clone()
		self.dmodel.zero()
		self.setDefaults()

	def reset(self):
		self.setDefaults()
		self.dmodel.zero()
		self.grad.zero()

	def objgradf(self, model):
		"""
		Computes Objective and Gradient distributed via Dask.
		"""
		# Map to Physical Domain
		if self.precond_op:
			self.precond_op.forward(False, model, self.phys_model)
		else:
			self.phys_model.copy(model)

		# Reset Physical Gradient
		self.phys_grad.zero()

		# Distributed FWI Calculation
		# We pass metadata to help Dask understand the return structure
		meta_df = pd.DataFrame({'norm_sq': pd.Series(dtype='float64'),
								'grad': pd.Series(dtype=object)})
		
		# map_partitions returns a Series of (obj, grad) per partition
		res_df = self.data.map_partitions(
			_obj_grad_worker,
			self.phys_model, 
			self.wavelet,
			self.prop_par,
			self.shots_per_gpu,       # Pass batch size
			self.gpu_stream_batches,
			self.geometry_mapping,
			meta=meta_df
		)

		# Aggregate Results
		fut_obj = res_df['norm_sq'].sum()
		fut_grad = res_df['grad'].reduction(
			chunk = _reduce_vector,
			aggregate = _reduce_vector,
			meta=pd.Series([],dtype=object)
		)
		total_obj, total_grad = dask.compute(fut_obj, fut_grad, retries=self.retry_tasks)
		self.obj = 0.5 * total_obj
		self.phys_grad.copy(total_grad.values[0])

		# Regularization (Physical Domain)
		if self.reg_op:
			self.reg_op.forward(False, self.phys_model, self.reg_vec)
			self.obj += 0.5 * (self.epsilon**2) * self.reg_vec.norm()**2
			
			# Grad_reg calculation
			self.reg_vec.scale(self.epsilon**2)
			self.reg_op.adjoint(True, self.phys_grad, self.reg_vec)

		# Apply Gradient Mask if Provided
		if self.grad_mask:
			self.phys_grad.multiply(self.grad_mask)

		# Backpropagate Gradient to Optimization Domain
		if self.precond_op:
			self.precond_op.adjoint(False, self.grad, self.phys_grad)
		else:
			self.grad.copy(self.phys_grad)

		self.obj_updated = True
		self.grad_updated = True
		return self.obj, self.grad

	def get_obj(self, model):
		self.set_model(model)
		if not self.obj_updated:
			# Map to Physical Domain
			if self.precond_op:
				self.precond_op.forward(False, model, self.phys_model)
			else:
				self.phys_model.copy(model)

			# Distributed FWI Calculation
			# We pass metadata to help Dask understand the return structure
			meta_df = pd.DataFrame({'norm_sq': pd.Series(dtype='float64')})
			
			# map_partitions returns a Series of (obj, grad) per partition
			res_df = self.data.map_partitions(
				_obj_worker,
				self.phys_model, 
				self.wavelet,
				self.prop_par,
				self.shots_per_gpu,       # Pass batch size
				self.gpu_stream_batches,
				self.geometry_mapping,
				meta=meta_df
			)

			# Aggregate Results
			total_df = res_df.sum().compute(retries=self.retry_tasks)
			self.obj = 0.5 * total_df['norm_sq']
		return self.obj

	def get_grad(self, model):
		self.set_model(model)
		if not self.grad_updated:
			_, self.grad = self.objgradf(model)
		return self.grad
	
	def get_rnorm(self, model):
		obj = self.get_obj(model)
		return np.sqrt(2.0 * obj)

	def get_res(self, model):
		raise NotImplementedError("FWIX residuals are too large for memory. Use get_obj() or get_rnorm().")
