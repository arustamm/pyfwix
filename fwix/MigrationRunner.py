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
from fwix.workers import _sum_gradients_on_disk, _io_load_and_migrate, safe_load_pickle
from fwix import CudaOperator as CuOp
from fwix.FWIXProblem import FWIXProblem

class MigrationRunner(FWIXProblem):
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
		
		self.client = get_client()
		if scratch_dir is None:
			raise ValueError("scratch_dir must be specified for FWIXProblem.")
		if not os.path.exists(scratch_dir):
			os.makedirs(scratch_dir, exist_ok=True)
		self.scratch_dir = scratch_dir
		
		self.data = data_pipeline.execute(return_pandas=False)

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
		# Default: Coarse Model is the same as Start Model (Physical)
		self.image = start_model.clone()

		self.grad_mask = problem_par.get("grad_mask", None)

	def run(self, model, save_path):
		
		# 1. Forward Mapping
		self.split_op.forward(False, model, self.phys_model)
		self.phys_grad.zero()

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
				task = dask.delayed(_io_load_and_migrate)(
					part_delayed, 
					target_path,     # <--- Passing String (Tiny)
					part_wav, 
					self.prop_par, 
					self.shots_per_gpu, 
					self.gpu_stream_batches, 
					self.geometry_mapping, 
					freq_id,
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
				f_id, f_paths = res
				
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

		self.split_op.adjoint(False, self.image, self.phys_grad)
		# the second component is just a ghost
		self.image.vecs[0].writeVec(save_path)

		gc.collect()

		return
	