import pandas as pd
import numpy as np
import gc
import pyVector as Vec
from typing import Tuple, Dict, Any
from fwix import CudaWEM
import genericIO
from pyZarrVector import ZarrVector

from fwix.utils import create_geometry, \
	create_wavelet, create_data, get_axis, zarr_to_sepvector, \
	get_slices

def _obj_grad_worker(
	df: pd.DataFrame,
	model: Vec.superVector, 	# ZarrVector
	wavelet: pd.DataFrame,
	prop_par,
	shots_per_gpu: int,
	gpu_stream_batches: Tuple[int],
	geom_mapping: Dict[str, str]
) -> pd.DataFrame:
	"""
	Worker function executed on every Dask partition.
	Loops over shots in micro-batches to compute obj and grad.
	"""
	
	# Initialize accumulators for this partition
	obj = 0.0
	grad = model.cloneSpace()
	shot_col = geom_mapping['id']
	unique_shots = df[shot_col].unique()
	padx = prop_par.get("ginsu_x", 0.0)
	pady = prop_par.get("ginsu_y", 0.0)

	# --- Micro-Batch Loop ---
	for i in range(0, len(unique_shots), shots_per_gpu):
		
		batch_ids = unique_shots[i : i + shots_per_gpu]
		df_batch = df[df[shot_col].isin(batch_ids)]
		wav_batch = wavelet.loc[batch_ids]
		
		# prepare geometry and model space
		geometry = create_geometry(df_batch, geom_mapping)
		slices = get_slices(geometry, model.vecs[0], padx, pady)
		local_slow = zarr_to_sepvector(model.vecs[0], slices=slices)
		local_den = zarr_to_sepvector(model.vecs[1], slices=slices)
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
			# Accumulate Objective
			obj += np.real(res.dot(res))
			
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
	return pd.DataFrame({"norm_sq": [obj], "grad": [grad]})


def _obj_worker(
	df: pd.DataFrame,
	model: Vec.superVector, 
	wavelet: pd.DataFrame,
	prop_par: Dict[str, Any],
	shots_per_gpu: int,
	gpu_stream_batches: Tuple[int],
	geom_mapping: Dict[str, str]
) -> pd.DataFrame:
	"""
	Worker function executed on every Dask partition.
	Loops over shots in micro-batches to compute objective.
	"""
	
	# Initialize accumulators for this partition
	obj = 0.0
	shot_col = geom_mapping['id']
	unique_shots = df[shot_col].unique()
	padx = prop_par.get("ginsu_x", 0.0)
	pady = prop_par.get("ginsu_y", 0.0)
	# --- Micro-Batch Loop ---
	prop = None
	for i in range(0, len(unique_shots), shots_per_gpu):
		
		batch_ids = unique_shots[i : i + shots_per_gpu]
		df_batch = df[df[shot_col].isin(batch_ids)]
		wav_batch = wavelet.loc[batch_ids]
		
		# prepare geometry and model space
		geometry = create_geometry(df_batch, geom_mapping)
		slices = get_slices(geometry, model.vecs[0], padx, pady)
		
		time_axis = get_axis(wav_batch)
		wav_vec = create_wavelet(wav_batch, time_axis)
		data = create_data(df_batch, time_axis)
		res = data.clone()

		local_slow = zarr_to_sepvector(model.vecs[0], slices=slices)
		local_den = zarr_to_sepvector(model.vecs[1], slices=slices)
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
			# Accumulate Objective
			obj += np.real(res.dot(res))

		finally:
			# Clean up GPU memory explicitly
			del prop, data, wav_vec, res
			gc.collect()

	# Return summary for this partition
	return pd.DataFrame({"norm_sq": [obj]})

def _reduce_vector(series):
	"""
	Safe Reducer: Uses streaming summation to prevent Zarr corruption.
	"""
	# 1. Extract list of valid vectors
	vectors = [v for v in series if v is not None]
	if not vectors:
		return None

	first_vec = vectors[0]

	# We must unzip them, sum the components, and zip them back.
	n_components = len(first_vec.vecs)
	summed_components = []

	for i in range(n_components):
		# Gather the i-th component from ALL vectors (e.g., all Slowness vectors)
		comp_list = [sv.vecs[i] for sv in vectors]
		
		# Use the static streaming sum (No intermediate files!)
		summed_comp = ZarrVector.sum(comp_list)
		summed_components.append(summed_comp)

	# Return new superVector
	return Vec.superVector(*summed_components)
