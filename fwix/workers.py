import pandas as pd
import numpy as np
import gc
import pyVector as Vec
from typing import Tuple, Dict, Any
from fwix import CudaWEM
import genericIO
from pyZarrVector import ZarrVector
import dask
import logging
import socket

from fwix.utils import create_geometry, \
	create_wavelet, create_data, get_axis, slice_sepvector, \
	get_slices

def _obj_grad_worker(
	df: pd.DataFrame,
	model: Vec.superVector, 	# ZarrVector
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
	
	logger = logging.getLogger("fwix.workers")
	logger.setLevel(logging.INFO) # Ensure we capture info
    
	# Add Worker ID to logs to find bad hardware
	worker_id = socket.gethostname()
	log_prefix = f"[{worker_id} | Freq {freq_id}]"

	# Initialize accumulators for this partition
	norm_sq = 0.0
	grad = model.clone().zero()
	shot_col = geom_mapping['id']
	unique_shots = df[shot_col].unique()
	padx = prop_par.get("ginsu_x", 0.0)
	pady = prop_par.get("ginsu_y", 0.0)

	# --- Micro-Batch Loop ---
	for i in range(0, len(unique_shots), shots_per_gpu):
		
		batch_ids = unique_shots[i : i + shots_per_gpu]
		df_batch = df[df[shot_col].isin(batch_ids)]
		wav_batch = wavelet.loc[(freq_id, list(batch_ids)), :]
		if np.isnan(df_batch['data'].iloc[0]).any():
			logger.error(f"{log_prefix} FATAL: Input Data contains NaN for shots {batch_ids}")
			return (freq_id, float('nan'), None)

		# prepare geometry and model space
		geometry = create_geometry(df_batch, geom_mapping)
		slices = get_slices(geometry, model.vecs[0], padx, pady)
		local_slow = slice_sepvector(model.vecs[0], slices)
		local_den = slice_sepvector(model.vecs[1], slices)
		local_model = Vec.superVector(local_slow, local_den)

		min_s, max_s = local_slow.min(), local_slow.max()
		if min_s <= 0 or np.isnan(min_s):
			logger.error(f"{log_prefix} FATAL: Bad Slowness in slice. Min: {min_s}, Max: {max_s}")
			return (freq_id, float('nan'), None)

		# update the padding
		prop_par["padx"] = local_slow.shape[-1]
		prop_par["pady"] = local_slow.shape[-2]

		time_axis = get_axis(wav_batch)
		wav_vec = create_wavelet(wav_batch, time_axis)
		wav_nrm = wav_vec.norm()
		if wav_nrm == 0 or np.isnan(wav_nrm):
			logger.error(f"{log_prefix} FATAL: Wavelet is silent or NaN. Norm: {wav_nrm}")
			return (freq_id, float('nan'), None)
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

			sim_norm = res.norm()
			if np.isnan(sim_norm) or np.isinf(sim_norm):
				logger.error(f"{log_prefix} EXPLOSION: Forward prop produced NaN/Inf. Shots: {batch_ids}")
				# Optional: dump bad model slice to disk here for inspection
				return (freq_id, float('nan'), None)

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
		
		time_axis = get_axis(wav_batch)
		wav_vec = create_wavelet(wav_batch, time_axis)
		data = create_data(df_batch, time_axis)
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
						compute_grad=True):
	"""
	Worker wrapper that loads the velocity model from Disk (Scratch)
	instead of receiving it over the network.
	"""
	# 1. Load the Velocity Model from the shared filesystem
	# genericIO handles reading SepVectors (.H files) efficiently
	slow = genericIO.defaultIO.getVector(model_path[0])
	den = genericIO.defaultIO.getVector(model_path[1])
	slow_den = Vec.superVector(slow, den)

    # 2. Run the original worker logic
	if compute_grad:
		return _obj_grad_worker(
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
	
