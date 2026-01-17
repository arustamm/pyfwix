import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
from dask.distributed import get_client, as_completed
import gc
from typing import Tuple, Dict, Any
from collections import defaultdict
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
import pickle

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
from fwix.workers import _io_load_and_compute, _sum_gradients_on_disk, _io_load_and_compute_born, safe_load_pickle
from fwix import CudaOperator as CuOp

class FWIXProblem(Prblm.Problem):
	def __init__(self, 
				start_model: Vec.superVector, 
				data_pipeline,
				prop_par: Dict[str, Any],
				wavelet: pd.DataFrame,
				problem_par: dict,
				shots_per_gpu: int = 1,
				gpu_stream_batches: Tuple[int] = (1, 1),
				geometry_mapping: Dict[str, str] = {
					"sx": "sx", "sy": "sy", "sz": "sz",
					"id": "uniqueshots",
					"rx": "rx", "ry": "ry", "rz": "rz",
					"freq_id" : "freq_band_id"
				},
				retry_tasks: int = 3,
				scratch_dir: str = None,
				n_freq_splits: int = None,
			):
		
		super(FWIXProblem, self).__init__()
		self.client = get_client()
		if scratch_dir is None:
			raise ValueError("scratch_dir must be specified for FWIXProblem.")
		if not os.path.exists(scratch_dir):
			os.makedirs(scratch_dir, exist_ok=True)
		self.scratch_dir = scratch_dir
		
		self.data = data_pipeline.execute(return_pandas=False)
		# self.data = self.data.persist()

		self.prop_par = prop_par
		self.problem_par = problem_par
		self.retry_tasks = retry_tasks

		freq_col = geometry_mapping['freq_id']
		shot_col = geometry_mapping['id']

		print("Mapping pure partitions...")
		self.partition_map = self._get_partition_map(self.data, freq_col)
		print(self.partition_map)
		
		# CREATE INVERTED MAP (compute once)
		self.partition_to_freq = self._invert_partition_map(self.partition_map)
	
		# wavelet_indexed = wavelet.drop_duplicates(subset=[freq_col, shot_col])
		wavelet_indexed = wavelet.set_index([freq_col, shot_col]).sort_index()
		self.wavelet = wavelet_indexed
		
		self.shots_per_gpu = shots_per_gpu
		self.gpu_stream_batches = gpu_stream_batches
		self.geometry_mapping = geometry_mapping

		# --- Build model, preconditioner, etc (keep your existing code) ---
		self.split_op = CuOp.SplitOperator(start_model, n_splits=n_freq_splits)
		self.phys_model = self.split_op.range.clone()
		self.phys_grad = self.phys_model.clone().zero()
		self.split_op.forward(False, start_model, self.phys_model)

		self._create_prec_op(start_model)
		self._create_prox_op()

		self.reg_op = None
		self.epsilon = problem_par.get("reg", {}).get("epsilon", 0.0)
		if self.epsilon > 0:
			self.reg_op = CuOp.Derivative(self.phys_model, self.phys_model, which=1, 
										  order=4, mode=problem_par["reg"]['mode'])
			self.reg_vec = self.phys_model.clone()

		self.grad = self.model.clone().zero()
		self.grad_mask = problem_par.get("grad_mask", None)
		self.dmodel = self.model.clone()
		self.dmodel.zero()
		self.setDefaults()

	def _create_prec_op(self, start_model):
		self.precond_op = None
		if 'pre' in self.problem_par:
			print("Building 4D Spline Preconditioner...")
			
			# Helper to build a coarse vector from a fine one
			def build_coarse(fine_vec, ns_coarse_dims):
				fine_hyper = fine_vec.getHyper()
				axes_coarse = []
				# Iterate 1..4 (SepVector Axis convention)
				for i in range(4):
					ax_fine = fine_hyper.getAxis(i + 1)
					n_c = ns_coarse_dims[i]
					d_c = (ax_fine.n - 1) * ax_fine.d / (n_c - 1)
					axes_coarse.append(Hypercube.axis(n=n_c, o=ax_fine.o, d=d_c))
				return SepVector.getSepVector(axes=axes_coarse, storage='dataComplex')

			# A. Geometry Definitions
			fine_slow = start_model.vecs[0] # 4D Fine Slowness
			fine_den  = start_model.vecs[1] # 4D Fine Density
			
			# Fetch config dictionary
			ns_config = self.problem_par['pre']['ns']
			
			# 1. Slowness Coarse Grid
			coarse_slow = build_coarse(fine_slow, ns_config['slow'])
			# 2. Density Coarse Grid
			coarse_den = build_coarse(fine_den, ns_config['den'])
			
			# Optimization Model is a SuperVector of these two potentially different grids
			self.model = Vec.superVector(coarse_slow, coarse_den)
			self.model.zero()
			
			# B. Create Operators
			# 1. Splines (Coarse 4D -> Fine 4D)
			# Ensure Spline4D is imported from your pyCudaOperator module
			op_spline_slow = CuOp.Spline4D(coarse_slow, fine_slow, type="CR-spline")
			op_spline_den  = CuOp.Spline4D(coarse_den,  fine_den,  type="CR-spline")
			
			# 2. Combine Splines (Parallel Dstack)
			# We explicitly pass the Domain (self.model) and Range (start_model)
			op_spline_combined = Op.Dstack([op_spline_slow, op_spline_den])
			
			# 3. Chain: Spline -> Split
			self.precond_op = Op.ChainOperator(op_spline_combined, self.split_op)
			
			print("Initializing Preconditioned Model via Linear CGLS...")
			LinStop  = Stopper.BasicStopper(niter=self.problem_par['pre']['niter'])
			CGsolver = LinearSolver.LCGsolver(LinStop)
			
			# Solve: Chain * m_coarse = phys_model
			# Note: The problem calculates: residuals = Op*m - d
			InitProb = Prblm.ProblemL2Linear(self.model, self.phys_model, self.precond_op)
			CGsolver.setDefaults(save_obj=False, save_res=False, save_grad=False, save_model=False)
			CGsolver.run(InitProb, verbose=True)
			
		else:
			self.model = start_model.clone()
			self.precond_op = self.split_op
		
	def _compute_bound_arrays(self, keys: list, comp_index: int, transform_func):
		"""
		Helper to compute lower and upper bounds for a specific model component.
		
		Args:
			keys: List [key_for_lower_bound, key_for_upper_bound] from problem_par
			comp_index: 0 for Slowness, 1 for Density
			transform_func: Function to convert par value to model unit (e.g. v -> 1/v^2)
		"""
		bound_arrays = []
		
		# Iterate twice: once for Lower Bound, once for Upper Bound
		for lim_key in keys:
			if lim_key not in self.problem_par:
				raise ValueError(f"Missing parameter key for bounds: {lim_key}")
				
			val = self.problem_par[lim_key]
			target_val = transform_func(val)

			# 1. Create Physical Target (SuperVector of Bands)
			phys_target = self.phys_model.clone()
			phys_target.zero()

			# Set the specific component (0 or 1) to target_val for all bands
			for band_vec in phys_target.vecs:
				# Zero out everything first (already done by clone().zero())
				# Set target component
				band_vec.vecs[comp_index].set(target_val)

			# 2. Project back to Optimization Model Space
			model_limit = self.model.clone()
			model_limit.zero()

			if self.precond_op:
				# If we have a preconditioner (Splines), we must invert it to find
				# what the coarse grid values should be to achieve these bounds.
				BoundProb = Prblm.ProblemL2Linear(model_limit, phys_target, self.precond_op)
				
				LinStop  = Stopper.BasicStopper(niter=self.problem_par.get('pre', {}).get('niter', 5))
				CGsolver = LinearSolver.LCGsolver(LinStop)
				CGsolver.setDefaults(save_obj=False, save_res=False, save_grad=False, save_model=False)
				CGsolver.run(BoundProb, verbose=True)
			else:
				# If no preconditioner, the mapping is direct (identity/split)
				# Just set the optimization model component directly
				model_limit.vecs[comp_index].set(target_val)

			# Extract the component array we care about
			bound_arrays.append(model_limit.vecs[comp_index].getNdArray())
			
		return bound_arrays[0], bound_arrays[1]

	def _create_prox_op(self):
		"""
		Creates Proximal Operator for the 4D Optimization Model.
		Model Structure: SuperVector([4D_Slowness, 4D_Density])
		"""
		self.proxOp = None
		
		# --- 1. Slowness Prox (Index 0) ---
		slow_prox = None
		if ("vmin" in self.problem_par) and ("vmax" in self.problem_par):
			print("Computing 4D Proximal Bounds for Slowness...")
			# Note order: vmax corresponds to LOWER bound of Slowness (1/v^2)
			#             vmin corresponds to UPPER bound of Slowness
			lower_s, upper_s = self._compute_bound_arrays(
				keys=["vmax", "vmin"], 
				comp_index=0, 
				transform_func=lambda v: 1.0 / (v**2)
			)
			# Use separate arrays to ensure PyProximal doesn't hold refs to reusable buffers
			slow_prox = ProxOperatorExplicit(
				pp.Box(lower=lower_s.copy(), upper=upper_s.copy())
			)

		# --- 2. Density Prox (Index 1) ---
		den_prox = None
		if ("rho_min" in self.problem_par) and ("rho_max" in self.problem_par):
			print("Computing 4D Proximal Bounds for Density...")
			# Density is direct: rho_min is Lower, rho_max is Upper
			lower_d, upper_d = self._compute_bound_arrays(
				keys=["rho_min", "rho_max"], 
				comp_index=1, 
				transform_func=lambda rho: rho
			)
			den_prox = ProxOperatorExplicit(
				pp.Box(lower=lower_d.copy(), upper=upper_d.copy())
			)

		# --- 3. Combine ---
		# ProxDstack applies prox operators to components of a SuperVector
		if slow_prox or den_prox:
			self.proxOp = ProxDstack([slow_prox, den_prox])

	def _get_partition_map(self, ddf, freq_col):
		def get_partition_freq(df):
			if len(df) == 0:
				return -1
			return int(df[freq_col].iloc[0])
		part_freqs = ddf.map_partitions(get_partition_freq).compute()
		p_map = defaultdict(list)
		for i, fid in enumerate(part_freqs):
			if fid != -1:
				p_map[fid].append(i)
		return dict(p_map)
	
	def _invert_partition_map(self, p_map):
		inv_map = {}
		for freq_id, part_indices in p_map.items():
			for pid in part_indices:
				inv_map[pid] = freq_id
		return inv_map

	def reset(self):
		self.setDefaults()
		self.dmodel.zero()
		self.grad.zero()

	def _write_models(self, tmp_dir, model):

		def _write_slice_worker(mod, idx, tmp_dir):
			"""
			Writes model slices to disk using Pickle.
			"""
			# Change extension to .pkl
			vel_name = os.path.join(tmp_dir, f"slow_freq_{idx}.pkl")
			den_name = os.path.join(tmp_dir, f"den_freq_{idx}.pkl")
			
			# Write Slowness
			with open(vel_name, 'wb') as f:
				pickle.dump(mod.vecs[0], f, protocol=pickle.HIGHEST_PROTOCOL)
				
			# Write Density
			with open(den_name, 'wb') as f:
				pickle.dump(mod.vecs[1], f, protocol=pickle.HIGHEST_PROTOCOL)
			
			return [vel_name, den_name]
		
		model_paths = [None] * len(model.vecs) # Pre-allocate list
		with ThreadPoolExecutor(max_workers=len(model_paths)) as executor:
			futures = {}
			for i, mod in enumerate(model.vecs):
				# Submit the write task
				future = executor.submit(_write_slice_worker, mod, i, tmp_dir)
				futures[future] = i
			
			# Collect results as they finish (or wait for all)
			for future in futures:
				i = futures[future]
				model_paths[i] = future.result()
				
		return model_paths

	def objgradf(self, model):
		# 1. Forward Mapping
		self.precond_op.forward(False, model, self.phys_model)
		self.phys_grad.zero()
		self.obj = 0.0
		# TEMP HACK: Ensure no positive imaginary parts
		mask = np.where(model[0][:].imag >=0 )
		model[0][mask].imag = 0.0

		# --- STEP 2: WRITE MODELS TO SCRATCH ---
		model_tmp_dir = tempfile.mkdtemp(dir=self.scratch_dir, prefix="fwix_models_")
		grad_tmp_dir = tempfile.mkdtemp(dir=self.scratch_dir, prefix="fwix_grad_")
		try:
			model_paths = self._write_models(model_tmp_dir, self.phys_model)
			
			# --- STEP 3: BUILD GRAPH ---
			delayed_partitions = self.data.to_delayed()
			inv_map = self._invert_partition_map(self.partition_map)
			task_futures = []
			ftag = self.geometry_mapping['freq_id']

			for part_idx, part_delayed in enumerate(delayed_partitions):
				freq_id = inv_map.get(part_idx)
				if freq_id is None: continue

				# KEY CHANGE: Pass the FILENAME string, not the object
				target_path = model_paths[freq_id]

				mask = self.wavelet.index.get_level_values(ftag) == freq_id
				part_wav = self.wavelet[mask]

				# Call the new IO Wrapper
				task = dask.delayed(_io_load_and_compute)(
					part_delayed, 
					target_path,     # <--- Passing String (Tiny)
					part_wav, 
					self.prop_par, 
					self.shots_per_gpu, 
					self.gpu_stream_batches, 
					self.geometry_mapping, 
					freq_id,
					compute_grad=True,
					grad_tmp_dir = grad_tmp_dir,
				)
				task_futures.append(task)

			# --- STEP 4: COMPUTE & WRITE ---
			freq_results = self.client.compute(task_futures, retries=self.retry_tasks)

			grad_files_map = defaultdict(lambda: defaultdict(list))
			
			for fut in as_completed(freq_results):
				res = fut.result()
				if res is None: continue
				
				# res is (f_id, f_obj, [path_comp0, path_comp1])
				f_id, f_obj, f_paths = res
				
				self.obj += 0.5 * f_obj
				
				# Organize by component
				for comp_idx, path in enumerate(f_paths):
					grad_files_map[f_id][comp_idx].append(path)

			# Lists to track our work
			reduction_futures = {} # future -> (f_id, comp_idx)
			ready_to_read = []     # list of (f_id, comp_idx, path)
			
			for f_id, comp_dict in grad_files_map.items():
				for comp_idx, paths in comp_dict.items():
					if not paths: continue

					# CASE A: Single File (Ready immediately)
					if len(paths) == 1:
						ready_to_read.append((f_id, comp_idx, paths[0]))
						
					# CASE B: Multiple Files (Needs Summing)
					else:
						sum_filename = f"summed_freq{f_id}_comp{comp_idx}.H"
						sum_path = os.path.join(grad_tmp_dir, sum_filename)
						
						# Fire and forget (don't wait!)
						future = self.client.submit(
							_sum_gradients_on_disk, 
							paths, 
							sum_path,
						)
						reduction_futures[future] = (f_id, comp_idx)

			# --- PIPELINE IO ---
			
			# Step 1: Read the single files NOW (While cluster computes sums)
			for f_id, comp_idx, path in ready_to_read:
				# Direct read
				vec = safe_load_pickle(path)
				self.phys_grad.vecs[f_id].vecs[comp_idx].scaleAdd(vec, 1.0, 1.0)
				del vec

			# Step 2: Read summation results as they arrive
			if reduction_futures:
				
				# as_completed yields futures as soon as they finish
				for future in as_completed(list(reduction_futures.keys())):
					sum_path = future.result()
					f_id, comp_idx = reduction_futures[future]
					
					# Read the summed result
					vec = safe_load_pickle(sum_path)
					self.phys_grad.vecs[f_id].vecs[comp_idx].scaleAdd(vec, 1.0, 1.0)
					del vec

		finally:
			# --- STEP 5: CLEANUP ---
			# This runs even if Dask crashes, keeping your scratch clean
			import shutil
			if os.path.exists(model_tmp_dir):
				shutil.rmtree(model_tmp_dir)
			if os.path.exists(grad_tmp_dir):
				shutil.rmtree(grad_tmp_dir)

		if self.reg_op:
			self.reg_op.forward(False, self.phys_model, self.reg_vec)
			self.obj += 0.5 * (self.epsilon**2) * self.reg_vec.norm()**2
			self.reg_vec.scale(self.epsilon**2)
			self.reg_op.adjoint(True, self.phys_grad, self.reg_vec)

		if self.grad_mask:
			self.phys_grad.multiply(self.grad_mask)

		self.precond_op.adjoint(False, self.grad, self.phys_grad)

		self.obj_updated = True
		self.grad_updated = True
		gc.collect()
		
		return self.obj, self.grad

	def get_obj(self, model):
		self.set_model(model)
		if self.obj_updated:
			return self.obj

		# 1. Forward Mapping (Model -> Physical)
		self.precond_op.forward(False, model, self.phys_model)
		self.obj = 0.0

		# 4. BUILD SCALAR GRAPH
		# We iterate the list of partitions purely in Python.
		obj_tasks = []
		ftag = self.geometry_mapping['freq_id']
		tmp_dir = tempfile.mkdtemp(dir=self.scratch_dir, prefix="fwix_obj_")
		try:
			model_paths = self._write_models(tmp_dir, self.phys_model)
			inv_map = self._invert_partition_map(self.partition_map)
			delayed_partitions = self.data.to_delayed()

			for part_idx, part_delayed in enumerate(delayed_partitions):
				# 1. Lookup Freq ID
				freq_id = inv_map.get(part_idx)
				if freq_id is None: continue
				
				target_path = model_paths[freq_id]
				
				mask = self.wavelet.index.get_level_values(ftag) == freq_id
				part_wav = self.wavelet[mask]

				# 3. Create Worker Task
				# Note: We use _obj_worker here (lighter), not _obj_grad_worker
				worker_task = dask.delayed(_io_load_and_compute)(
					part_delayed, 
					target_path, 
					part_wav, 
					self.prop_par, 
					self.shots_per_gpu, 
					self.gpu_stream_batches, 
					self.geometry_mapping, 
					freq_id,
					compute_grad=False
				)
				obj_tasks.append(worker_task)

			# 5. COMPUTE & SUM
			obj_results = self.client.compute(obj_tasks, retries=self.retry_tasks)
			
			# Accumulate results as they arrive
			for fut in as_completed(obj_results):
				val = fut.result()
				self.obj += 0.5 * val
		finally:
			import shutil
			if os.path.exists(tmp_dir):
				shutil.rmtree(tmp_dir)

			# 6. REGULARIZATION
		if self.reg_op:
			self.reg_op.forward(False, self.phys_model, self.reg_vec)
			self.obj += 0.5 * (self.epsilon**2) * self.reg_vec.norm()**2

		self.obj_updated = True
		return self.obj
	
	def dresresf(self, model, dmodel):
		"""
		Calculates dot products (res . dres) and (dres . dres) distributedly
		without gathering the massive residual vectors.
		"""
		
		# Apply preconditioner if it exists:
		# Physical Model m = P * model
		# Physical Search Direction dm = P * dmodel
		phys_dmodel = self.phys_model.clone()
		self.precond_op.forward(False, model, self.phys_model)
		self.precond_op.forward(False, dmodel, phys_dmodel)

		total_res_dres = 0.0
		total_dres_dres = 0.0

		# 2. Write BOTH vectors to scratch
		# We need the workers to access both m and dm
		model_tmp_dir = tempfile.mkdtemp(dir=self.scratch_dir, prefix="fwix_models_")
		dmod_tmp_dir = tempfile.mkdtemp(dir=self.scratch_dir, prefix="fwix_dmod_")
		
		try:
			model_paths = self._write_models(model_tmp_dir, self.phys_model)
			dmodel_paths = self._write_models(dmod_tmp_dir, phys_dmodel)

			# 3. Build Dask Graph
			delayed_partitions = self.data.to_delayed()
			inv_map = self._invert_partition_map(self.partition_map)
			tasks = []
			ftag = self.geometry_mapping['freq_id']

			for part_idx, part_delayed in enumerate(delayed_partitions):
				freq_id = inv_map.get(part_idx)
				if freq_id is None: continue

				m_path = model_paths[freq_id]
				dm_path = dmodel_paths[freq_id]

				mask = self.wavelet.index.get_level_values(ftag) == freq_id
				part_wav = self.wavelet[mask]

				# Submit task
				task = dask.delayed(_io_load_and_compute_born)(
					part_delayed, 
					m_path, 
					dm_path,
					part_wav, 
					self.prop_par, 
					self.shots_per_gpu, 
					self.gpu_stream_batches, 
					self.geometry_mapping, 
					freq_id
				)
				tasks.append(task)

			# 4. Compute and Sum Results
			# results will be a list of tuples: [(res_dres, dres_dres), ...]
			results = self.client.compute(tasks)
			
			for fut in as_completed(results):
				r_d, d_d = fut.result()
				total_res_dres += r_d
				total_dres_dres += d_d

		finally:
			import shutil
			if os.path.exists(model_tmp_dir):
				shutil.rmtree(model_tmp_dir)
			if os.path.exists(dmod_tmp_dir):
				shutil.rmtree(dmod_tmp_dir)

		# 5. Add Regularization Terms (Analytical)
		# res_reg = epsilon * Reg * m
		# dres_reg = epsilon * Reg * dm
		if self.reg_op:
			reg_dres = self.reg_vec.clone()

			# Apply Reg Operator
			self.reg_op.forward(False, self.phys_model, self.reg_vec)
			self.reg_op.forward(False, phys_dmodel, reg_dres)
			
			# Scale by epsilon
			self.reg_vec.scale(self.epsilon)
			reg_dres.scale(self.epsilon)

			# Add to dot products
			total_res_dres += self.reg_vec.dot(reg_dres)
			total_dres_dres += reg_dres.dot(reg_dres)

		del phys_dmodel
		gc.collect()

		return total_res_dres, total_dres_dres

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