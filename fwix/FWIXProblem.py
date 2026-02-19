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

		self.objective = problem_par.get('obj', 'l2')
		print(f"FWIXProblem initialized with objective: {self.objective}")

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
		# Default: Coarse Model is the same as Start Model (Physical)
		self.model = start_model.clone()

		self._build_operators(start_model, n_freq_splits)

		self.reg_op = None
		self.epsilon = problem_par.get("epsilon", 0.0)
		if self.epsilon > 0:
			print(f"-> Adding DSO Regularization with epsilon={self.epsilon}")
			self.reg_op = CuOp.Derivative(self.phys_model, self.phys_model)
			self.reg_vec = self.phys_model.clone()

		self.grad = self.model.clone().zero()
		self.grad_mask = problem_par.get("grad_mask", None)
		self.dmodel = self.model.clone()
		self.dmodel.zero()
		self.setDefaults()

	def _build_operators(self, start_model, n_freq_splits):
		"""
		Constructs the self.precond_op chain and initializes self.model
		handling all combinations of 'pre' and 'bounds'.
		"""
		print("--- Building Optimization Chain ---")

		# 1. BASE: The Physical Split Operator
		# This always exists.
		self.split_op = CuOp.SplitOperator(start_model, n_splits=n_freq_splits)
		self.phys_model = self.split_op.range.clone()
		self.phys_grad = self.phys_model.clone().zero()
		self.split_op.forward(False, start_model, self.phys_model)

		# Current "Head" of the chain (starts at physical input)
		# As we add operators, this 'current_op' grows backwards.
		current_op = self.split_op
		
		# The model input to the current operator
		# Starts as the Fine Grid Physical Model
		current_model = start_model.clone()

		# 2. OPTIONAL: Linear Preconditioner (Splines)
		# If 'pre' is in parameters, we define Coarse Grid -> Fine Grid
		if 'pre' in self.problem_par:
			print("-> Adding Linear Preconditioner (Splines)")
			
			# A. Define Coarse Grid
			fine_slow = start_model.vecs[0]
			fine_den  = start_model.vecs[1]
			ns_config = self.problem_par['pre']['ns']
			
			# Helper to build coarse axes
			def build_coarse_vec(fine_vec, dims):
				fine_hyper = fine_vec.getHyper()
				axes_coarse = []
				for i in range(4): # 4D
					ax_fine = fine_hyper.getAxis(i + 1)
					n_c = dims[i]
					d_c = (ax_fine.n - 1) * ax_fine.d / (n_c - 1) if n_c > 1 else ax_fine.d
					axes_coarse.append(Hypercube.axis(n=n_c, o=ax_fine.o, d=d_c))
				return SepVector.getSepVector(axes=axes_coarse, storage='dataComplex')

			coarse_slow = build_coarse_vec(fine_slow, ns_config['slow'])
			coarse_den  = build_coarse_vec(fine_den, ns_config['den'])
			
			# This becomes our new "current_model"
			coarse_model = Vec.superVector(coarse_slow, coarse_den)
			coarse_model.zero()

			# B. Create Operators
			op_spline_slow = CuOp.Spline4D(coarse_slow, fine_slow, type="CR-spline")
			op_spline_den  = CuOp.Spline4D(coarse_den,  fine_den,  type="CR-spline")
			op_spline_combined = Op.Dstack([op_spline_slow, op_spline_den])

			# C. Invert to initialize Coarse Model (Optional but recommended)
			print("   Initializing Coarse Model values...")
			LinStop  = Stopper.BasicStopper(niter=self.problem_par['pre'].get('init_iter', 5))
			CGsolver = LinearSolver.LCGsolver(LinStop)
			# Solve: Spline * m_coarse = m_fine
			InitProb = Prblm.ProblemL2Linear(coarse_model, current_model, op_spline_combined)
			CGsolver.setDefaults(save_obj=False, save_res=False, save_grad=False, save_model=False)
			CGsolver.run(InitProb, verbose=False)

			# D. Chain: Spline -> [Current Chain]
			# Note: Op.ChainOperator(Op1, Op2) applies Op2(Op1(x))
			# We want Split(Spline(x)), so Chain(Spline, Split)
			current_op = Op.ChainOperator(op_spline_combined, current_op)
			
			# Update current model pointer
			current_model = coarse_model

		# 3. OPTIONAL: Bounds (SoftMinMax)
		# If bounds exist, we wrap the current model (whether Coarse or Fine)
		if any(x in self.problem_par for x in ["vmin", "vmax", "rho_min", "rho_max"]):
			print("-> Adding Soft Constraints (Bounds)")
			
			# A. Prepare Bound Values
			# Note: We pass the 'current_op' to project bounds to the correct grid (Coarse or Fine)
			slow_max, slow_min = None, None
			if "vmin" in self.problem_par:
				slow_max, slow_min = self._compute_bound_arrays(
					["vmin", "vmax"], 0, lambda v: 1./v**2, 
					current_model, current_op # <-- Pass current state
				)

			den_min, den_max = None, None
			if "rho_min" in self.problem_par:
				den_min, den_max = self._compute_bound_arrays(
					["rho_min", "rho_max"], 1, lambda r: r,
					current_model, current_op
				)

			# B. Create Unbounded Model (The Domain)
			unbounded_model = current_model.clone()
			
			# C. Create Soft Operators
			# They map Unbounded -> Bounded (Current Model)
			tau = self.problem_par.get('tau', 0.001)
			
			slow_op = CuOp.SoftMinMax(
				model=unbounded_model.vecs[0], 
				data=current_model.vecs[0], # Target is the constrained model
				xmin=slow_min, xmax=slow_max, tau=[tau, tau]
			)
			den_op = CuOp.SoftMinMax(
				model=unbounded_model.vecs[1], 
				data=current_model.vecs[1], 
				xmin=den_min, xmax=den_max, tau=[tau, tau]
			)

			# D. Stack and Chain
			nl_dstack = Op.Dstack([slow_op.nl_op, den_op.nl_op])
			lin_dstack = Op.Dstack([slow_op.lin_op, den_op.lin_op])
			bound_op = Op.NonLinearOperator(nl_dstack, lin_dstack)

			# Wrap current linear op as nonlinear to chain it
			if not isinstance(current_op, Op.NonLinearOperator):
				current_op = Op.NonLinearOperator(current_op, current_op)
			
			# Chain: Linear( Bound ( x ) )
			current_op = Op.CombNonlinearOp(bound_op, current_op)
			
			# Update current model pointer
			current_model = unbounded_model

			if self.problem_par.get("enforce_neg_imag", True):
				print("-> Adding Imaginary-Part Constraint (HyperbolicPenalty)")
				
				tau = self.problem_par.get('tau', 0.001)
				
				# Instantiate your monolithic operators
				# Note: Domain = unbounded_imag, Range = current_model (which might be real-bounded)
				imag_nl  = CuOp.HyperbolicPenalty(current_model, current_model, l=1.0, tau=tau)
				imag_lin = CuOp.Softclip(current_model, current_model, l=1.0, tau=tau)
				
				imag_bound_op = Op.NonLinearOperator(imag_nl, imag_lin)
				
				# Chain: ImagBound -> [Current]
				if not isinstance(current_op, Op.NonLinearOperator):
					current_op = Op.NonLinearOperator(current_op, current_op)
					
				current_op = Op.CombNonlinearOp(imag_bound_op, current_op)

		# 4. FINALIZE
		self.precond_op = current_op
		self.model = current_model
		print("--- Optimization Chain Built ---")

	def _compute_bound_arrays(self, keys, comp_index, transform_func, model_template, mapping_op):
		"""
		Computes bound arrays on the grid (Coarse or Fine).
		Args:
			model_template: The vector space where bounds should live (Coarse or Fine).
			mapping_op: The operator that maps model_template -> Physical.
		"""
		results = []
		
		# Helper Solver
		LinStop  = Stopper.BasicStopper(niter=10)
		CGsolver = LinearSolver.LCGsolver(LinStop)
		CGsolver.setDefaults(save_obj=False, save_res=False, save_grad=False, save_model=False)

		for key in keys:
			if key not in self.problem_par:
				# Handle missing keys (e.g. if only vmin provided but not vmax)
				results.append(None)
				continue

			val = self.problem_par[key]
			target_val = transform_func(val)
			
			# 1. Create Physical Target (Constant value)
			phys_target = self.phys_model.clone()
			phys_target.zero()
			for band_vec in phys_target.vecs:
				band_vec.vecs[comp_index].set(target_val)

			# 2. Solve for the Projection
			# We solve: mapping_op * m_grid = phys_target
			grid_target = model_template.clone()
			grid_target.zero()
			
			# If mapping_op is just SplitOp (Identity grid), projection is trivial
			if isinstance(mapping_op, CuOp.SplitOperator):
				# Direct set (copy) since grid is same
					# (Logic slightly complex due to split, but Solver handles it generally)
					pass 

			print(f"Projecting bound {key}={val}...")
			BoundProb = Prblm.ProblemL2Linear(grid_target, phys_target, mapping_op)
			CGsolver.run(BoundProb, verbose=False)
			
			bound_array = grid_target.vecs[comp_index].getNdArray().real
			results.append(bound_array)

		return results[0], results[1]
	
	def set_model(self, model):
		"""
		Updates the model and triggers the Non-Linear chain update.
		"""
		super().set_model(model)
		if self.precond_op:
			self.precond_op.set_background(model)

	# def _create_prox_op(self):
	# 	"""
	# 	Creates Proximal Operator for the 4D Optimization Model.
	# 	Model Structure: SuperVector([4D_Slowness, 4D_Density])
	# 	"""
	# 	self.proxOp = None
		
	# 	# --- 1. Slowness Prox (Index 0) ---
	# 	slow_prox = None
	# 	if ("vmin" in self.problem_par) and ("vmax" in self.problem_par):
	# 		print("Computing 4D Proximal Bounds for Slowness...")
	# 		# Note order: vmax corresponds to LOWER bound of Slowness (1/v^2)
	# 		#             vmin corresponds to UPPER bound of Slowness
	# 		lower_s, upper_s = self._compute_bound_arrays(
	# 			keys=["vmax", "vmin"], 
	# 			comp_index=0, 
	# 			transform_func=lambda v: -2.0 * np.log(v)
	# 		)
	# 		# Use separate arrays to ensure PyProximal doesn't hold refs to reusable buffers
	# 		slow_prox = ProxOperatorExplicit(
	# 			pp.Box(lower=lower_s.copy(), upper=upper_s.copy())
	# 		)

	# 	# --- 2. Density Prox (Index 1) ---
	# 	den_prox = None
	# 	if ("rho_min" in self.problem_par) and ("rho_max" in self.problem_par):
	# 		print("Computing 4D Proximal Bounds for Density...")
	# 		# Density is direct: rho_min is Lower, rho_max is Upper
	# 		lower_d, upper_d = self._compute_bound_arrays(
	# 			keys=["rho_min", "rho_max"], 
	# 			comp_index=1, 
	# 			transform_func=lambda rho: np.log(rho)
	# 		)
	# 		den_prox = ProxOperatorExplicit(
	# 			pp.Box(lower=lower_d.copy(), upper=upper_d.copy())
	# 		)

	# 	# --- 3. Combine ---
	# 	# ProxDstack applies prox operators to components of a SuperVector
	# 	if slow_prox or den_prox:
	# 		self.proxOp = ProxDstack([slow_prox, den_prox])

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
	
	def _compute_regularization(self, compute_grad=True):
		"""
		Computes regularization objective and gradient on the Head Node.
		Safe to run while Dask cluster is processing data.
		"""
		reg_obj = 0.0
		if self.reg_op:
			# 1. Compute Reg Objective
			self.reg_op.forward(False, self.phys_model, self.reg_vec)
			# Norm squared of complex vector: sum(|x|^2)
			reg_obj = 0.5 * (self.epsilon**2) * self.reg_vec.norm()**2

			if compute_grad:
				# 2. Compute Reg Gradient
				# Scale residual by epsilon^2 before adjoint
				self.reg_vec.scale(self.epsilon**2)
				
				# Accumulate directly into phys_grad
				self.reg_op.adjoint(True, self.phys_grad, self.reg_vec)
				
		return reg_obj

	def objgradf(self, model):
		
		# 1. Forward Mapping
		self.precond_op.forward(False, model, self.phys_model)
		self.phys_grad.zero()
		self.obj_terms = [0.0,0,0]  # data, reg

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
					obj_type = self.objective,
				)
				task_futures.append(task)

			# --- STEP 4: COMPUTE & WRITE ---
			freq_results = self.client.compute(task_futures, retries=self.retry_tasks)

			# While cluster computes, accumulate regularization
			self.obj_terms[1] = self._compute_regularization(compute_grad=True)

			grad_files_map = defaultdict(lambda: defaultdict(list))
			for fut in as_completed(freq_results):
				res = fut.result()
				if res is None: continue
				
				# res is (f_id, f_obj, [path_comp0, path_comp1])
				f_id, f_obj, f_paths = res
				
				self.obj_terms[0] += 0.5 * f_obj
				
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

		if self.grad_mask:
			self.phys_grad.multiply(self.grad_mask)

		zpow = self.problem_par.get("zpow", 0.0)
		if zpow != 0.0:
			# Iterate over Frequency Bands
			for band_vec in self.phys_grad.vecs:
				# Iterate over Components (Slow, Den)
				for comp_vec in band_vec.vecs:
					nz = comp_vec.getHyper().getAxis(4).n					
					weights = np.linspace(1, nz, nz)
					weights = np.power(weights, zpow).reshape(-1,1,1,1)
					comp_vec[:] *= weights

		self.precond_op.jac.adjoint(False, self.grad, self.phys_grad)

		self.obj_updated = True
		self.grad_updated = True
		gc.collect()
		
		return sum(self.obj_terms), self.grad

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
					compute_grad=False,
					obj_type = self.objective
				)
				obj_tasks.append(worker_task)

			# 5. COMPUTE & SUM
			obj_results = self.client.compute(obj_tasks, retries=self.retry_tasks)

			# While cluster computes, accumulate regularization
			self.obj += self._compute_regularization(compute_grad=False)
			
			# Accumulate results as they arrive
			for fut in as_completed(obj_results):
				val = fut.result()
				self.obj += 0.5 * val
		finally:
			import shutil
			if os.path.exists(tmp_dir):
				shutil.rmtree(tmp_dir)

		self.fevals += 1
		self.obj_updated = True
		return self.obj
	
	def dresresf(self, model, dmodel):
		"""
		Calculates dot products (res . dres) and (dres . dres) distributedly
		without gathering the massive residual vectors.
		"""
		self.set_model(model)
		# Apply preconditioner if it exists:
		# Physical Model m = P * model
		# Physical Search Direction dm = P * dmodel
		phys_dmodel = self.phys_model.clone()
		self.precond_op.forward(False, model, self.phys_model)
		self.precond_op.jac.forward(False, dmodel, phys_dmodel)

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
					freq_id,
					obj_type = self.objective,
				)
				tasks.append(task)

			# 4. Compute and Sum Results
			# results will be a list of tuples: [(res_dres, dres_dres), ...]
			results = self.client.compute(tasks)

			if self.reg_op:
				reg_dres = self.reg_vec.clone()

				# A. Regularization Residual: r_reg = D * m
				self.reg_op.forward(False, self.phys_model, self.reg_vec)
				
				# B. Linearized Reg Residual: dr_reg = D * dm
				self.reg_op.forward(False, phys_dmodel, reg_dres)
				
				# C. Compute Dot Products
				# Term 1: (epsilon * r_reg) . (epsilon * dr_reg) = eps^2 * (r_reg . dr_reg)
				reg_dot_prod = self.reg_vec.dot(reg_dres)
				total_res_dres += (self.epsilon**2) * reg_dot_prod
				
				# Term 2: (epsilon * dr_reg) . (epsilon * dr_reg) = eps^2 * (dr_reg . dr_reg)
				reg_norm_sq = reg_dres.dot(reg_dres)
				total_dres_dres += (self.epsilon**2) * reg_norm_sq
			
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