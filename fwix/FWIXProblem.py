import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
from dask.distributed import get_client, as_completed, worker_client
import gc
from typing import Tuple, Dict, Any
from collections import defaultdict
import os
import tempfile

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
from fwix.workers import _io_load_and_compute, _obj_worker, _build_tree_reduction
from fwix import CudaOperator

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
				scratch_dir: str = None
			):
		
		super(FWIXProblem, self).__init__()
		self.client = get_client()
		if scratch_dir is None:
			raise ValueError("scratch_dir must be specified for FWIXProblem.")
		if not os.path.exists(scratch_dir):
			os.makedirs(scratch_dir, exist_ok=True)
		self.scratch_dir = scratch_dir
		
		self.data = data_pipeline.execute(return_pandas=False)
		self.data = self.data.persist()

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
	
		wavelet_indexed = wavelet.drop_duplicates(subset=[freq_col, shot_col])
		wavelet_indexed = wavelet_indexed.set_index([freq_col, shot_col]).sort_index()
		self.wavelet = wavelet_indexed
		
		self.shots_per_gpu = shots_per_gpu
		self.gpu_stream_batches = gpu_stream_batches
		self.geometry_mapping = geometry_mapping

		# --- Build model, preconditioner, etc (keep your existing code) ---
		self.phys_model = start_model
		self.phys_grad = self.phys_model.clone().zero()
		self.precond_op = None
		
		if 'pre' in problem_par:
			print("Building Preconditioner...")
			ops_pre = []
			vecs_pre = []
			for i, mod in enumerate(self.phys_model.vecs):
				ax = mod.getHyper().axes
				ns_pre = problem_par['pre']['ns'][i]
				ds_pre = [(ax[j].n-1)*ax[j].d / (ns_pre[j] - 1) for j in range(len(ns_pre))]
				mod_pre = SepVector.getSepVector(
					Hypercube.hypercube(ns=ns_pre, ds=ds_pre, os=[a.o for a in ax]), 
					storage='dataComplex'
				)
				mod_pre.zero()
				vecs_pre.append(mod_pre)
				interp = CudaOperator.Spline4D(mod_pre, mod, type="CR-spline")
				ops_pre.append(interp)
			
			self.model = Vec.superVector(vecs_pre[0], vecs_pre[1])
			self.precond_op = Op.Dstack(self.model, self.phys_model, ops_pre)
			
			print("Initializing Preconditioned Model via Linear CGLS...")
			LinStop  = Stopper.BasicStopper(niter=problem_par['pre']['niter'])
			CGsolver = LinearSolver.LCGsolver(LinStop)
			InitProb = Prblm.ProblemL2Linear(self.model, self.phys_model, self.precond_op)
			CGsolver.setDefaults(save_obj=False, save_res=False, save_grad=False, save_model=False)
			CGsolver.run(InitProb, verbose=True)
		else:
			self.model = self.phys_model.clone()
			self.precond_op = None

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

		self.reg_op = None
		self.epsilon = problem_par.get("reg", {}).get("epsilon", 0.0)
		if self.epsilon > 0:
			self.reg_op = CudaOperator.Derivative(self.phys_model, self.phys_model, which=1, 
										  order=4, mode=problem_par["reg"]['mode'])
			self.reg_vec = self.phys_model.clone()

		self.grad = self.model.clone().zero()
		self.grad_mask = problem_par.get("grad_mask", None)
		self.dmodel = self.model.clone()
		self.dmodel.zero()
		self.setDefaults()

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

	def _write_models(self, tmp_dir):
		model_paths = []
		# Write each frequency slice to disk
		for i, mod in enumerate(self.phys_model.vecs):
			vel_name = os.path.join(tmp_dir, f"slow_freq_{i}.H")
			den_name = os.path.join(tmp_dir, f"den_freq_{i}.H")
			# Write superVector
			mod.vecs[0].writeVec(vel_name)
			mod.vecs[1].writeVec(den_name)
			model_paths.append([vel_name, den_name])  # Store paths as list of [vel_path, den_path]
		return model_paths

	def objgradf(self, model):
		# 1. Forward Mapping
		if self.precond_op:
			self.precond_op.forward(False, model, self.phys_model)
		else:
			self.phys_model.copy(model)
		
		self.phys_grad.zero()
		self.obj = 0.0

		# --- STEP 2: WRITE MODELS TO SCRATCH ---
		tmp_dir = tempfile.mkdtemp(dir=self.scratch_dir, prefix="fwix_models_")
		
		try:
			model_paths = self._write_models(tmp_dir)
			
			# --- STEP 3: BUILD GRAPH ---
			delayed_partitions = self.data.to_delayed()
			inv_map = self._invert_partition_map(self.partition_map)
			freq_groups = defaultdict(list)
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
					freq_id
				)
				freq_groups[freq_id].append(task)

			# --- STEP 4: REDUCTION (Unchanged) ---
			freq_accumulators = []
			for freq_id, tasks in freq_groups.items():
				if not tasks: continue
				freq_tree_sum = _build_tree_reduction(tasks)
				if freq_tree_sum is not None:
					freq_accumulators.append(freq_tree_sum)

			freq_results = self.client.compute(freq_accumulators, retries=self.retry_tasks)

			for fut in as_completed(freq_results):
				result = fut.result()
				if result is None: continue
				# Unpack (Tuple is guaranteed now)
				f_id, f_obj, f_grad = result
				
				self.obj += 0.5 * f_obj
				if f_grad is not None:
					self.phys_grad.vecs[f_id].scaleAdd(f_grad, 1.0, 1.0)
					del f_grad 

		finally:
			# --- STEP 5: CLEANUP ---
			# This runs even if Dask crashes, keeping your scratch clean
			import shutil
			if os.path.exists(tmp_dir):
				shutil.rmtree(tmp_dir)

		# --- Regularization & Backprop (Unchanged) ---
		if self.reg_op:
			self.reg_op.forward(False, self.phys_model, self.reg_vec)
			self.obj += 0.5 * (self.epsilon**2) * self.reg_vec.norm()**2
			self.reg_vec.scale(self.epsilon**2)
			self.reg_op.adjoint(True, self.phys_grad, self.reg_vec)

		if self.grad_mask:
			self.phys_grad.multiply(self.grad_mask)

		if self.precond_op:
			self.precond_op.adjoint(False, self.grad, self.phys_grad)
		else:
			self.grad.copy(self.phys_grad)

		self.obj_updated = True
		self.grad_updated = True
		return self.obj, self.grad

	def get_obj(self, model):
		self.set_model(model)
		if self.obj_updated:
			return self.obj

		# 1. Forward Mapping (Model -> Physical)
		if self.precond_op:
			self.precond_op.forward(False, model, self.phys_model)
		else:
			self.phys_model.copy(model)
		
		self.obj = 0.0


		# 4. BUILD SCALAR GRAPH
		# We iterate the list of partitions purely in Python.
		obj_tasks = []
		ftag = self.geometry_mapping['freq_id']
		tmp_dir = tempfile.mkdtemp(dir=self.scratch_dir, prefix="fwix_obj_")
		try:
			model_paths = self._write_models(tmp_dir)
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