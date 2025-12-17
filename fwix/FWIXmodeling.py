import pandas as pd
import numpy as np
import dask.dataframe as dd
from pysep3d.step import ReaderStep 
import SepVector
from typing import Dict, List, Any, Tuple
import pyarrow as pa
import gc
from fwix import CudaWEM
import Hypercube
import genericIO
from fwix.utils import create_geometry, create_wavelet, create_data, get_axis

class FWIXmodeling(ReaderStep):
    def __init__(self, 
                model: Any, 
                wavelet: pd.DataFrame, 
                prop_par: Dict[str, Any],
                geometry: Dict[str, Any], 
                partition_size: int,   # Dask: Number of shots per Dask partition
                shots_per_gpu: int = 1, # GPU: Number of shots per C++ batch
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
				}	
            ):
        
        self.geometry = geometry
        self.gpu_stream_batches = gpu_stream_batches
        self.partition_size = partition_size
        self.shots_per_gpu = shots_per_gpu
        self.model = model
        self.prop_par = prop_par
        self.geometry_mapping = geometry_mapping

        # --- 1. Identify Frequency Bands from Wavelet ---
        freq_col = geometry_mapping['freq_id']
        shot_col = geometry_mapping['id']

        # Ensure wavelet has the frequency column
        if freq_col not in wavelet.columns:
            raise ValueError(f"Wavelet dataframe must contain column '{freq_col}'")

        # Get unique frequencies and sort them
        unique_freqs = np.unique(wavelet[freq_col].values)
        
        base_pdf = self._create_trace_headers(geometry)
        
        # We replicate the shot/receiver geometry for EACH frequency band
        dfs = []
        for fid in unique_freqs:
            tmp = base_pdf.copy()
            tmp[freq_col] = fid
            dfs.append(tmp)
        
        # Combine and sort to ensure efficient partitioning (group by Freq, then Shot)
        full_pdf = pd.concat(dfs, ignore_index=True)
        if not full_pdf[shot_col].is_monotonic_increasing:
             full_pdf = full_pdf.sort_values([freq_col, shot_col])

        # We index by (Freq, Shot) for fast lookup in the worker
        # Dropping duplicates just in case, though your prep should handle this
        wavelet_indexed = wavelet.drop_duplicates(subset=[freq_col, shot_col])
        wavelet_indexed = wavelet_indexed.set_index([freq_col, shot_col]).sort_index()
        self.wavelet = wavelet_indexed

        # Partition size applies to 'Unique Tasks' (Freq + Shot pairs)
        # We count unique (freq, shot) combinations to determine partitions
        unique_tasks = full_pdf[[freq_col, shot_col]].drop_duplicates()
        n_tasks = len(unique_tasks)
        npartitions = int(np.ceil(n_tasks / self.partition_size))
        
        self.df = dd.from_pandas(full_pdf, npartitions=npartitions)
        
        # Metadata for Dask
        self.meta = full_pdf.iloc[:0].copy()
        self.meta['data'] = pd.Series(dtype=object)

    def _create_trace_headers(self, geom: Dict[str, Any]) -> pd.DataFrame:
        """
        Expands compact geometry (lists of unique source/rec locs) 
        into a full trace table (one row per trace).
        """        
        unique_sx = np.atleast_1d(geom['sx'])
        unique_sy = np.atleast_1d(geom['sy'])
        unique_sz = np.atleast_1d(geom['sz'])
        
        n_shots = len(unique_sx)
        n_rcvs = len(geom['rx']) / n_shots
        
        # s_ids: [0, 0, 0, ..., 1, 1, 1, ...]
        shots = np.repeat(np.arange(n_shots), n_rcvs)
        
        sx = np.repeat(unique_sx, n_rcvs)
        sy = np.repeat(unique_sy, n_rcvs)
        sz = np.repeat(unique_sz, n_rcvs)
        
        rx = geom['rx']
        ry = geom['ry']
        rz = geom['rz']
        
        df = pd.DataFrame({
            'uniqueshots': shots.astype(np.int32),
            'sx': sx.astype(np.float32), 'sy': sy.astype(np.float32), 'sz': sz.astype(np.float32),
            'rx': rx, 'ry': ry, 'rz': rz,
        })
        return df

    def create(self) -> dd.DataFrame:
        simulated_ddf = self.df.map_partitions(
            _simulate_partition,
            self.model,
            self.wavelet,
            self.prop_par,
            self.shots_per_gpu,
            self.gpu_stream_batches,
            self.geometry_mapping,
            meta=self.meta
        )
        return simulated_ddf.reset_index()
    

def _simulate_partition(
    df: pd.DataFrame, 
    model, 
    wavelet: pd.DataFrame,
    prop_par: Dict[str, Any],
    shots_per_gpu: int,
    gpu_stream_batches: tuple,
    geom_mapping: Dict[str, str]
) -> pd.DataFrame:
    
    freq_col = geom_mapping['freq_id']
    shot_col = geom_mapping['id']
    
    # 1. Sort locally to ensure we process Freqs -> Shots sequentially
    # This is crucial so we can fill 'data_list' in order
    df = df.sort_values([freq_col, shot_col])
    
    # Prepare list to hold results
    data_list = [None] * len(df)
    current_row_offset = 0
    
    # 2. Outer Loop: Frequency Bands
    # We group by frequency to minimize model slicing/switching
    unique_freqs = np.sort(df[freq_col].unique())
    
    for fid in unique_freqs:
        # Filter dataframe for this band
        df_band = df[df[freq_col] == fid]
        
        # SLICE MODEL: Grab the specific band [[Vel_i, Den_i]] from SuperVector
        # Assumes model.vecs is ordered corresponding to freq IDs (0, 1, 2...)
        model_band = model.vecs[fid]

        unique_shots = df_band[shot_col].unique()
        
        # 3. Inner Loop: Micro-Batches of Shots
        for i in range(0, len(unique_shots), shots_per_gpu):
            
            batch_ids = unique_shots[i : i + shots_per_gpu]
            df_batch = df_band[df_band[shot_col].isin(batch_ids)]
            ntraces_batch = len(df_batch)

            # SLICE WAVELET: MultiIndex lookup (FreqID, ShotIDs)
            # Tuple key format: (fid, [list_of_shots])
            # We use .loc carefully to handle MultiIndex
            try:
                wav_df = wavelet.loc[(fid, list(batch_ids)), :]
                # .loc might drop the index levels, verify 'data' column exists
                if isinstance(wav_df, pd.Series): 
                    # Handle edge case of single shot returning Series
                    wav_df = wav_df.to_frame().T
            except KeyError:
                 raise KeyError(f"Wavelet not found for Freq {fid} and Shots {batch_ids}")

            geometry = create_geometry(df_batch, geom_mapping)
            time_axis = get_axis(wav_df)
            wav_vec = create_wavelet(wav_df, time_axis)
            data = create_data(df_batch, time_axis)

            prop_par["padx"] = model_band.vecs[0].shape[-1]
            prop_par["pady"] = model_band.vecs[0].shape[-2]
            par = genericIO.pythonParams(prop_par)
            prop = None
            try:
                prop = CudaWEM.Propagator(
                    model_band, # Pass the specific band model
                    data, wav_vec, 
                    par, geometry, nbatches=gpu_stream_batches
                )
                prop.forward(False, model_band, data)
                
                # Fill the results list sequentially
                # Since we sorted 'df' at the start, iterating freq->shot matches the list order
                # Note: list(data[:]) converts the SepVector into a list of numpy arrays (traces)
                data_list[current_row_offset : current_row_offset + ntraces_batch] = list(data[:].copy())              
                current_row_offset += ntraces_batch

            finally:
                del prop, data, wav_vec
                gc.collect()

    # Create a copy to return
    res_df = df.copy()
    res_df['data'] = data_list
    
    return res_df