import pandas as pd
import numpy as np
import gc
import pyVector as Vec
from typing import Tuple, Dict, Any
from fwix import CudaWEM
import genericIO
from dask.distributed import print
import dask
import uuid
import os

from fwix.utils import create_geometry, \
	create_wavelet, create_data, get_axis, slice_sepvector, \
	get_slices

def _obj_grad_worker(
	df: pd.DataFrame,
	model: Vec.superVector, 	
	wavelet: pd.DataFrame,
	prop_par,
	shots_per_gpu: int,
	gpu_stream_batches: Tuple[int],
	geom_mapping: Dict[str, str],
	freq_id: int
) -> pd.DataFrame:
	"""
	Worker function executed on every Dask partition.
	Loops over shots in micro-batches to compute obj and grad.
	"""
	if len(df) == 0:
		return pd.DataFrame({'norm_sq': [], 'grad': [], 'freq_id': []})

	# Initialize accumulators for this partition
	norm_sq = 0.0
	grad = model.clone().zero()
	shot_col = geom_mapping['id']

	df = df.sort_values(shot_col)
	wavelet = wavelet.sort_index()

	unique_shots = df[shot_col].unique()
	padx = prop_par.get("ginsu_x", 0.0)
	pady = prop_par.get("ginsu_y", 0.0)

	# --- Micro-Batch Loop ---
	for i in range(0, len(unique_shots), shots_per_gpu):
		
		batch_ids = unique_shots[i : i + shots_per_gpu]
		df_batch = df[df[shot_col].isin(batch_ids)]
		wav_batch = wavelet.loc[(freq_id, list(batch_ids)), :]

		# prepare geometry and model space
		geometry = create_geometry(df_batch, geom_mapping)
		slices = get_slices(geometry, model.vecs[0], padx, pady)
		local_slow = slice_sepvector(model.vecs[0], slices)
		local_den = slice_sepvector(model.vecs[1], slices)
		local_model = Vec.superVector(local_slow, local_den)

		# update the padding
		prop_par["padx"] = local_slow.shape[-1]
		prop_par["pady"] = local_slow.shape[-2]

		time_axis = get_axis(wav_batch)
		wav_vec = create_wavelet(wav_batch, time_axis)
		data = create_data(df_batch, time_axis)
		
		res = data.clone()
		local_grad = local_model.clone().zero()
		par = genericIO.pythonParams(prop_par)

		try:
			prop = CudaWEM.Propagator(
				local_model, res, wav_vec, 
				par, geometry, nbatches=gpu_stream_batches
			)
			born = CudaWEM.ExtendedBorn(local_grad, res, local_model, prop)
			
			# Forward Modeling: d_sim = F(m)
			prop.forward(False, local_model, res) 

			# Compute Residual: r = F(m) - d_obs
			res.scaleAdd(data, 1.0, -1.0)
			# Accumulate norm_sqective
			norm_sq += np.real(res.dot(res))
			
			# Adjoint (Gradient): g = F' * r 
			born.adjoint(False, local_grad, res)
	
			# Accumulate Gradient into Global Grad
			grad.vecs[0][slices] += local_grad.vecs[0][:]	# slowness
			grad.vecs[1][slices] += local_grad.vecs[1][:]	# density

		finally:
			# Clean up GPU memory explicitly
			del prop, born, data, wav_vec, res
			gc.collect()

	# Return summary for this partition
	return (freq_id, float(norm_sq), grad)


def _obj_worker(
	df: pd.DataFrame,
	model: Vec.superVector, 
	wavelet: pd.DataFrame,
	prop_par: Dict[str, Any],
	shots_per_gpu: int,
	gpu_stream_batches: Tuple[int],
	geom_mapping: Dict[str, str],
	freq_id: int
) -> pd.DataFrame:
	"""
	Worker function executed on every Dask partition.
	Loops over shots in micro-batches to compute objective.
	"""
	
	# Initialize accumulators for this partition
	norm_sq = 0.0
	shot_col = geom_mapping['id']

	df = df.sort_values(shot_col)
	wavelet = wavelet.sort_index()

	unique_shots = df[shot_col].unique()
	padx = prop_par.get("ginsu_x", 0.0)
	pady = prop_par.get("ginsu_y", 0.0)
	# --- Micro-Batch Loop ---
	prop = None
	for i in range(0, len(unique_shots), shots_per_gpu):
		
		batch_ids = unique_shots[i : i + shots_per_gpu]
		df_batch = df[df[shot_col].isin(batch_ids)]
		wav_batch = wavelet.loc[(freq_id, list(batch_ids)), :]
		
		# prepare geometry and model space
		geometry = create_geometry(df_batch, geom_mapping)
		slices = get_slices(geometry, model.vecs[0], padx, pady)
		
		freq_axis = get_axis(wav_batch)
		wav_vec = create_wavelet(wav_batch, freq_axis)
		data = create_data(df_batch, freq_axis)
		res = data.clone()
		if wav_vec.norm() == 0:
			raise ValueError("Wavelet vector has zero norm.")
		if data.norm() == 0:
			raise ValueError("Data vector has zero norm.")

		local_slow = slice_sepvector(model.vecs[0], slices)
		local_den = slice_sepvector(model.vecs[1], slices)
		local_model = Vec.superVector(local_slow, local_den)

		prop_par["padx"] = local_slow.shape[-1]
		prop_par["pady"] = local_slow.shape[-2]
		par = genericIO.pythonParams(prop_par)
		
		try:
			prop = CudaWEM.Propagator(
				local_model, res, wav_vec, 
				par, geometry, nbatches=gpu_stream_batches
			)
			
			# Forward Modeling: d_sim = F(m)
			prop.forward(False, local_model, res) 
			# Compute Residual: r = F(m) - d_obs
			res.scaleAdd(data, 1.0, -1.0)
			# Accumulate norm_sqective
			norm_sq += np.real(res.dot(res))

		finally:
			# Clean up GPU memory explicitly
			del prop, data, wav_vec, res
			gc.collect()

	# Return summary for this partition
	return norm_sq

def _build_tree_reduction(delayed_items):
    if not delayed_items:
        return None

    # If 1 item, it's ALREADY a tuple from the worker. Return it directly.
    if len(delayed_items) == 1:
        return delayed_items[0]

    while len(delayed_items) > 1:
        new_level = []
        for i in range(0, len(delayed_items), 2):
            left = delayed_items[i]
            if i + 1 < len(delayed_items):
                right = delayed_items[i+1]
                summ = dask.delayed(_extract_and_sum)(left, right)
                new_level.append(summ)
            else:
                new_level.append(left)
        delayed_items = new_level
    
    return delayed_items[0]


def _extract_and_sum(item1, item2):
    """
    Simpler Reducer.
    Input: Tuples (freq_id, obj, grad) or None.
    Output: Tuple (freq_id, obj, grad).
    """
    # 1. Handle None (empty partitions)
    if item1 is None: return item2
    if item2 is None: return item1

    # 2. Unpack directly (No parsing needed!)
    id1, obj1, grad1 = item1
    id2, obj2, grad2 = item2

    if id1 != id2:
        raise ValueError(f"Freq ID mismatch: {id1} vs {id2}")

    # 3. Accumulate Gradient
    # Reuse grad1 memory if available
    if grad1 is not None and grad2 is not None:
        grad1.scaleAdd(grad2, 1.0, 1.0)
    elif grad1 is None:
        grad1 = grad2

    return (id1, obj1 + obj2, grad1)

def _io_load_and_compute(part_df, model_path, part_wav, prop_par, 
						shots_per_gpu, stream_batches, geom_map, freq_id,
						compute_grad=True, grad_tmp_dir=None):
	"""
	Worker wrapper that loads the velocity model from Disk (Scratch)
	instead of receiving it over the network.
	"""
	# 1. Load the Velocity Model from the shared filesystem
	slow = safe_load_pickle(model_path[0])
	den = safe_load_pickle(model_path[1])
	slow_den = Vec.superVector(slow, den)

    # 2. Run the original worker logic
	if not compute_grad:
		return _obj_worker(
			part_df, 
			slow_den, 
			part_wav, 
			prop_par, 
			shots_per_gpu, 
			stream_batches, 
			geom_map, 
			freq_id
		)
	
	else:
		if grad_tmp_dir is None:
			raise ValueError("grad_tmp_dir must be specified when compute_grad = True.")
		
		freq_id, norm_sq, grad = _obj_grad_worker(
								part_df, 
								slow_den, 
								part_wav, 
								prop_par, 
								shots_per_gpu, 
								stream_batches, 
								geom_map, 
								freq_id
							)
		out_paths = []
		unique_suffix = uuid.uuid4().hex
		# Loop over components (e.g., 0=Slow, 1=Den)
		for i, comp_vec in enumerate(grad.vecs):
			# Create explicit filename for this component
			fname = f"grad_freq{freq_id}_comp{i}_{unique_suffix}.pkl"
			path = os.path.join(grad_tmp_dir, fname)
			
			# Write just this component
			with open(path, 'wb') as f:
				pickle.dump(comp_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
			out_paths.append(path)

		# Cleanup
		del grad, slow, den
		
		# Return LIST of paths: [path_slow, path_den]
		return (freq_id, norm_sq, out_paths)
		
def _sum_gradients_on_disk(file_list, output_path):
	"""
	Reads a list of .H files and sums them into one.
	"""
	if not file_list:
		return None
		
	# Read the first file to initialize sum
	total_vec = safe_load_pickle(file_list[0])

	# Loop and accumulate
	# Efficient strategy: Read into a temp buffer and add
	for path in file_list[1:]:
		temp_vec = safe_load_pickle(path)
		total_vec.scaleAdd(temp_vec, 1.0, 1.0)
		
	# Write the final result
	with open(output_path, 'wb') as f:
		pickle.dump(total_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
	return output_path
	
def _born_worker(
	df: pd.DataFrame,
	model: Vec.superVector,     # Background model (m)
	dmodel: Vec.superVector,    # Search direction (dm)
	wavelet: pd.DataFrame,
	prop_par: Dict[str, Any],
	shots_per_gpu: int,
	gpu_stream_batches: Tuple[int],
	geom_mapping: Dict[str, str],
	freq_id: int
	) -> Tuple[float, float]:
	"""
	Computes (res . born_data) and (born_data . born_data) for a partition.
	"""
	if len(df) == 0:
		return 0.0, 0.0

	dot_res_dres = 0.0
	dot_dres_dres = 0.0

	shot_col = geom_mapping['id']
	df = df.sort_values(shot_col)
	wavelet = wavelet.sort_index()

	unique_shots = df[shot_col].unique()

	# Slice parameters
	padx = prop_par.get("ginsu_x", 0.0)
	pady = prop_par.get("ginsu_y", 0.0)

	try:
		for i in range(0, len(unique_shots), shots_per_gpu):
			batch_ids = unique_shots[i : i + shots_per_gpu]
			df_batch = df[df[shot_col].isin(batch_ids)]
			wav_batch = wavelet.loc[(freq_id, list(batch_ids)), :]

			# 1. Geometry and Slicing
			geometry = create_geometry(df_batch, geom_mapping)
			slices = get_slices(geometry, model.vecs[0], padx, pady)
			
			# Slice Background Model
			local_slow = slice_sepvector(model.vecs[0], slices)
			local_den = slice_sepvector(model.vecs[1], slices)
			local_model = Vec.superVector(local_slow, local_den)

			# Slice Search Direction (dmodel)
			local_dslow = slice_sepvector(dmodel.vecs[0], slices)
			local_dden = slice_sepvector(dmodel.vecs[1], slices)
			local_dmodel = Vec.superVector(local_dslow, local_dden)

			# 2. Setup Vectors
			time_axis = get_axis(wav_batch)
			wav_vec = create_wavelet(wav_batch, time_axis)
			data = create_data(df_batch, time_axis)
			
			# Vectors for results
			res = data.clone()  # Non-linear modeled data (will become residual)
			dres = data.clone() # Linearized modeled data (Born response)

			# Update padding for CudaWEM
			prop_par["padx"] = local_slow.shape[-1]
			prop_par["pady"] = local_slow.shape[-2]
			par = genericIO.pythonParams(prop_par)

			try:
				# 3. Initialize Propagator
				prop = CudaWEM.Propagator(
					local_model, res, wav_vec, 
					par, geometry, nbatches=gpu_stream_batches
				)
				# Initialize Born Operator
				# Born takes: (model_pert, data_pert, background_model, propagator)
				born = CudaWEM.ExtendedBorn(local_dmodel, dres, local_model, prop)

				# 4. Compute Non-Linear Residual: r = F(m) - d_obs
				prop.forward(False, local_model, res)
				res.scaleAdd(data, 1.0, -1.0) 

				# 5. Compute Linearized Residual: dr = J * dm
				born.forward(False, local_dmodel, dres)

				# 6. Accumulate Dot Products (Real parts only)
				dot_res_dres += res.dot(dres)
				dot_dres_dres += dres.dot(dres)

			finally:
				del prop, born, data, wav_vec, res, dres, local_model, local_dmodel
				gc.collect()
	except Exception as e:
		import traceback
		print(f"WORKER ERROR (Freq {freq_id}): {str(e)}")
		traceback.print_exc()
		raise e

	return dot_res_dres, dot_dres_dres

def _io_load_and_compute_born(part_df, model_path, dmodel_path, part_wav, prop_par, 
                              shots_per_gpu, stream_batches, geom_map, freq_id):
    """
    IO Wrapper: Loads both background model and search direction from scratch.
    """
    # Load Background
    slow = safe_load_pickle(model_path[0])
    den = safe_load_pickle(model_path[1])
    model = Vec.superVector(slow, den)

    # Load Search Direction
    dslow = safe_load_pickle(dmodel_path[0])
    dden = safe_load_pickle(dmodel_path[1])
    dmodel = Vec.superVector(dslow, dden)

    # Run computation
    res_dres, dres_dres = _born_worker(
        part_df, model, dmodel, part_wav, prop_par,
        shots_per_gpu, stream_batches, geom_map, freq_id
    )

    del model, dmodel, slow, den, dslow, dden
    return res_dres, dres_dres

import time
import pickle
def safe_load_pickle(filepath, retries=3):
	# 1. Wait for file to appear (Metadata Consistency)
	for i in range(retries):
		if os.path.exists(filepath):
			break
		time.sleep(1.0)
		
	if not os.path.exists(filepath):
		# Debug info if missing
		parent = os.path.dirname(filepath)
		try:
			print(f"DEBUG: {parent} contents: {os.listdir(parent)}")
		except:
			print(f"DEBUG: {parent} does not exist.")
		raise FileNotFoundError(f"Worker could not find pickle file: {filepath}")

	# 2. Load
	try:
		with open(filepath, 'rb') as f:
			vec = pickle.load(f)
		return vec
	except EOFError:
		# Handle partial writes (rare but possible if client crashes)
		raise RuntimeError(f"Corrupt (empty) pickle file: {filepath}")
	except Exception as e:
		raise RuntimeError(f"Failed to load pickle {filepath}: {e}")