import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
from dask.distributed import get_client, as_completed, wait
import gc
from typing import Tuple, Dict, Any
from collections import defaultdict

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
from fwix.workers import _obj_grad_worker, _obj_worker

from fwix import CudaOperator

class FWIXProblem(Prblm.Problem):
	def __init__(self, 
				start_model: Vec.superVector, 
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
					"freq_id" : "freq_band_id"
				},
				retry_tasks: int = 3,
			):
		
		super(FWIXProblem, self).__init__()
		self.client = get_client()
		# Data and Parameters
		# create the reading pipeline (i.e., read + float_to_complex conversion)
		self.data = data_pipeline.execute(return_pandas=False) # dask dataframe
		self.prop_par = prop_par
		self.problem_par = problem_par
		self.retry_tasks = retry_tasks

		
		freq_col = geometry_mapping['freq_id']
		shot_col = geometry_mapping['id']

		print("Mapping pure partitions...")
		self.partition_map = self._get_partition_map(
			self.data, freq_col
		)
	
		wavelet_indexed = wavelet.drop_duplicates(subset=[freq_col, shot_col])
		wavelet_indexed = wavelet_indexed.set_index([freq_col, shot_col]).sort_index()
		self.wavelet = wavelet_indexed
		
		# Batching Config (Crucial for Memory)
		self.shots_per_gpu = shots_per_gpu
		self.gpu_stream_batches = gpu_stream_batches
		self.geometry_mapping = geometry_mapping

		# --- 1. Build Physical Model Space (High Res) ---
		self.phys_model = start_model
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

	def _get_partition_map(self, ddf, freq_col):
		"""
		Returns {freq_id: [list_of_partition_indices]}
		Assumes partitions are homogeneous (pure).
		"""
		# 1. Light function: Get frequency of just the first row
		def get_partition_freq(df):
			if len(df) == 0:
				return -1 # Empty partition marker
			return int(df[freq_col].iloc[0])

		# 2. Compute map cheaply
		# Returns a list of ints, e.g., [0, 0, 0, 1, 1, 2, 2...]
		part_freqs = ddf.map_partitions(get_partition_freq).compute()

		# 3. Invert list to dict
		p_map = defaultdict(list)
		for i, fid in enumerate(part_freqs):
			if fid != -1:
				p_map[fid].append(i)
		
		return dict(p_map)

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
		self.obj = 0.0

		# Distributed FWI Calculation
		# We pass metadata to help Dask understand the return structure
		meta_df = pd.DataFrame({'norm_sq': pd.Series(dtype='float64'),
								'grad': pd.Series(dtype=object),
								'freq_id' : pd.Series(dtype=int)
								})
		
		all_obj_grad = []
		ftag = self.geometry_mapping['freq_id']
		# Schedule all computation
		# phys_model = [[Slow_0, Den_0], [Slow_1, Den_1], ...]
		for freq_id, slow_den in enumerate(self.phys_model.vecs):
			# Get the frequency slice of the data 
			indices = self.partition_map.get(freq_id, [])
			if not indices:
				continue 
			# 2. Select the subset of partitions directly
			# This creates a Dask DataFrame containing ONLY the pure partitions for this freq
			part_data = self.data.partitions[indices]
	
			mask = self.wavelet.index.get_level_values(ftag) == freq_id
			part_wav = self.wavelet[mask]
			part_df = part_data.map_partitions(
				_obj_grad_worker,
				slow_den, 
				part_wav,	# this is a dataframe
				self.prop_par,
				self.shots_per_gpu,       # Pass batch size
				self.gpu_stream_batches,
				self.geometry_mapping,
				freq_id,
				meta=meta_df
			)
			fut_res = self.client.compute(part_df.to_delayed(), retries=self.retry_tasks)
			all_obj_grad.extend(fut_res)

		# Gather all the sources chunks
		for fut_res, res_df in as_completed(all_obj_grad, with_results=True):
			if len(res_df) == 0:
				continue
			
			r_obj = res_df['norm_sq'].iloc[0]
			r_grad = res_df['grad'].iloc[0]
			r_freq = int(res_df['freq_id'].iloc[0])

			self.obj += r_obj * 0.5
			if not self.phys_grad.vecs[r_freq].checkSame(r_grad):
				raise ValueError("Gradient vector from worker does not match expected frequency gradient vector.")
			self.phys_grad.vecs[r_freq].scaleAdd(r_grad)
			del res_df

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

			self.obj = 0.0

			# Map to Physical Domain
			if self.precond_op:
				self.precond_op.forward(False, model, self.phys_model)
			else:
				self.phys_model.copy(model)

			# Distributed FWI Calculation
			# We pass metadata to help Dask understand the return structure
			meta_df = pd.DataFrame({'norm_sq': pd.Series(dtype='float64')})
			
			all_obj = []
			ftag = self.geometry_mapping['freq_id']
			# Schedule all computation
			# phys_model = [[Slow_0, Den_0], [Slow_1, Den_1], ...]
			
			for freq_id, slow_den in enumerate(self.phys_model.vecs):
				# Get the frequency slice of the data 
				indices = self.partition_map.get(freq_id, [])
				if not indices:
					continue 
				# 2. Select the subset of partitions directly
				# This creates a Dask DataFrame containing ONLY the pure partitions for this freq
				part_data = self.data.partitions[indices]
		
				mask = self.wavelet.index.get_level_values(ftag) == freq_id
				part_wav = self.wavelet[mask]
				part_df = part_data.map_partitions(
					_obj_worker,
					slow_den, 
					part_wav,	# this is a dataframe
					self.prop_par,
					self.shots_per_gpu,       # Pass batch size
					self.gpu_stream_batches,
					self.geometry_mapping,
					freq_id,
					meta=meta_df
				)
				fut_res = self.client.compute(part_df.to_delayed(), retries=self.retry_tasks)
				all_obj.extend(fut_res)

			# Gather all the sources chunks
			for fut_res, res_df in as_completed(all_obj, with_results=True):
				if len(res_df) == 0:
					continue

				r_obj = res_df['norm_sq'].iloc[0]
				self.obj += r_obj * 0.5

				del res_df

		self.obj_updated = True
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
