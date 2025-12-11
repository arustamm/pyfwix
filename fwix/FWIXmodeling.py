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
from fwix.utils import create_geometry, create_wavelet, create_data, get_axis

class FWIXmodeling(ReaderStep):
    def __init__(self, 
                model: Any, 
                wavelet: pd.DataFrame, 
                par: Any, 
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
				}	
            ):
        
        super().__init__(path="")
        
        self.geometry = geometry
        self.gpu_stream_batches = gpu_stream_batches
        self.partition_size = partition_size
        self.shots_per_gpu = shots_per_gpu
        self.model = model
        
        self.par = par
        self.geometry_mapping = geometry_mapping

        # This creates a DataFrame with N_shots * N_receivers rows
        pdf = self._create_trace_headers(geometry)
        shot_col = 'uniqueshots'

        if not pdf[shot_col].is_monotonic_increasing:
            pdf = pdf.sort_values(shot_col)
        if not wavelet[shot_col].is_monotonic_increasing:
            wavelet = wavelet.sort_values(shot_col)
            
        wavelet = wavelet.set_index(shot_col)

        # partition based on Unique Shots to ensure a shot isn't split across files
        n_unique_shots = len(pdf[shot_col].unique())
        npartitions = int(np.ceil(n_unique_shots / self.partition_size))
        
        self.df = dd.from_pandas(pdf, npartitions=npartitions)
        self.wavelet = wavelet
        
        self.meta = pdf.iloc[:0].copy()
        self.meta['data'] = pd.Series(dtype=object)

    def _create_trace_headers(self, geom: Dict[str, Any]) -> pd.DataFrame:
        """
        Expands compact geometry (lists of unique source/rec locs) 
        into a full trace table (one row per trace).
        """        
        unique_sx = np.atleast_1d(geom['sx'])
        unique_sy = np.atleast_1d(geom['sy'])
        unique_sz = np.atleast_1d(geom['sz'])
        
        unique_rx = np.atleast_1d(geom['rx'])
        unique_ry = np.atleast_1d(geom['ry'])
        unique_rz = np.atleast_1d(geom['rz'])
        
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
            self.par,
            self.shots_per_gpu,
            self.gpu_stream_batches,
            self.geometry_mapping,
            meta=self.meta
        )
        return simulated_ddf.reset_index()
    

def _simulate_partition(
    df: pd.DataFrame, # Partition containing N shots (where N = shots_per_partition)
    model, 
    wavelet: pd.DataFrame,
    par,
    shots_per_gpu: int, # Limit for Micro-batching
    gpu_stream_batches: tuple,
    geom_mapping: Dict[str, str]
) -> pd.DataFrame:
    
    shot_col = geom_mapping['id']
    unique_shots = df[shot_col].unique()
    # Prepare list to hold results
    data_list = [None] * len(df)
    
    current_row_offset = 0
    
    # Iterate over Micro-Batches
    for i in range(0, len(unique_shots), shots_per_gpu):
        
        batch_ids = unique_shots[i : i + shots_per_gpu]
        df_batch = df[df[shot_col].isin(batch_ids)]
        ntraces_batch = len(df_batch)

        wav_df = wavelet.loc[batch_ids]
        
        geometry = create_geometry(df_batch, geom_mapping)
        
        time_axis = get_axis(wav_df)
        wav_vec = create_wavelet(wav_df, time_axis)
        data = create_data(df_batch, time_axis)

        prop = None
        try:
            prop = CudaWEM.Propagator(
                model, data, wav_vec, 
                par, geometry, nbatches=gpu_stream_batches
            )
            prop.forward(False, model, data)
            
            data_list[current_row_offset:current_row_offset+ntraces_batch] = list(data[:].copy())              
            current_row_offset += ntraces_batch

        finally:
            del prop, data
            gc.collect()

    # Create a copy to return (Dask requires this to avoid side effects)
    res_df = df.copy()
    res_df['data'] = data_list
    
    return res_df