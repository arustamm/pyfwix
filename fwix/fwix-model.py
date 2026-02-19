#!/usr/bin/env python3
"""
Run 2D FWIX forward modeling using a JSON configuration file.
Supports frequency band splitting for multi-gpu/distributed execution.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import SepVector
import genericIO
from pyVector import superVector
import pyVector as Vec
from pysep3d import ComplexToFloat, DaskPipeline, PyArrowWriter
from fwix import FWIXmodeling
import dask_util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FWIX modeling via JSON config.")
    parser.add_argument("config", help="Path to the JSON configuration file.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        config = json.load(f)
    
    defaults = {
        "partition_size": 8,
        "shots_per_gpu": 8,
        "gpu_batches": [1, 2],
        "nref": 3,
        "eps": 0.04,
        "padx": 0.0,
        "pady": 0.0,
        "taperx": 0,
        "tapery": 0,
        "ref_look_ahead": 2,
        "compress_error": 1e-6,
        "wflds_to_store": 2,
        "wfld_path": os.environ.get("SCRATCH", "/tmp"),
        "shot_col": "uniqueshots",
        "sx_col": "sx", "sy_col": "sy", "sz_col": "sz",
        "rx_col": "rx", "ry_col": "ry", "rz_col": "rz",
        "default_y": 0.0,
        "num_freq_splits": 1
    }
    
    for key, val in defaults.items():
        config.setdefault(key, val)
        
    return config


def load_model_split(cfg: Dict[str, Any]):
    """
    Loads 2D model (Complex, Slowness^2), expands to 4D, and splits into frequency bands.
    """
    path = cfg['model']
    nf = cfg['nf']
    of = cfg['of']
    df = cfg['df']
    num_splits = cfg['num_freq_splits']
    
    # 1. Load 2D Model
    # Expects genericIO vector where axes[0]=X (Fast) and axes[1]=Z (Slow)
    model2d = genericIO.defaultIO.getVector(path)
    axes = model2d.getHyper().axes
    
    nx, nz = axes[0].n, axes[1].n
    dx, dz = axes[0].d, axes[1].d
    ox, oz = axes[0].o, axes[1].o
    
    # 2. Get Numpy View
    # Note: genericIO/SepVector returns numpy array in C-order: (Slowest, ..., Fastest)
    # Therefore, shape is (nz, nx)
    arr_xz = model2d[:]

    if arr_xz.shape != (nz, nx):
        raise ValueError(f"Model shape mismatch. Expected (nz={nz}, nx={nx}), got {arr_xz.shape}")

    # User specification: Model is already Slowness Squared and Complex.
    # No inversion (1/v^2) needed.
    slow_xz = arr_xz

    # 3. Split into Frequency Bands
    band_supervectors = []
    n_band_samples = nf // num_splits
    
    for i in range(num_splits):
        # A. Calculate Indices
        idx_start = i * n_band_samples
        idx_end = nf if i == num_splits - 1 else idx_start + n_band_samples
        current_nf = idx_end - idx_start
        current_of = of + (idx_start * df)
        
        # B. Define Band Geometry (Fast -> Slow: [x, y, f, z])
        band_ns = [nx, 1, current_nf, nz]
        band_ds = [dx, 1.0, df, dz]
        band_os = [ox, 0.0, current_of, oz]
        
        s_band = SepVector.getSepVector(ns=band_ns, ds=band_ds, os=band_os, storage='dataComplex')
        d_band = SepVector.getSepVector(ns=band_ns, ds=band_ds, os=band_os, storage='dataComplex')
        
        # C. Fill Data
        # s_band.getNdArray() returns shape (nz, current_nf, 1, nx)
        s_arr = s_band.getNdArray()
        d_arr = d_band.getNdArray()
        
        # D. Broadcasting
        # Input 'slow_xz' is (nz, nx). 
        # Target 's_arr' is (nz, nf, 1, nx).
        # We assume the model is constant across Frequency (axis 1) and Y (axis 2).
        # Reshape (nz, nx) -> (nz, 1, 1, nx) so it broadcasts correctly.
        s_arr[:] = slow_xz[:, None, None, :]
        d_arr[:] = 1.0 + 0j
        
        # E. Create Mini-SuperVector
        band_sv = Vec.superVector(s_band, d_band)
        band_supervectors.append(band_sv)
        
    # 4. Return Final Model
    if num_splits == 1:
        return band_supervectors[0]
    
    return Vec.superVector(*band_supervectors)


def _normalize_wavelet_array(vec) -> np.ndarray:
    arr = np.asarray(vec[:])
    if arr.ndim == 1: return arr[np.newaxis, :]
    if arr.ndim == 2:
        nt = vec.getHyper().axes[0].n
        if arr.shape[1] == nt: return arr
        if arr.shape[0] == nt: return arr.T
    raise ValueError(f"Invalid wavelet shape {arr.shape}")


def build_wavelet_df_split(cfg: Dict[str, Any], nshots: int) -> pd.DataFrame:
    wv = genericIO.defaultIO.getVector(cfg['wavelet'])
    axes = wv.getHyper().axes
    nt, dt = axes[0].n, axes[0].d
    
    nf = cfg['nf']
    of = cfg['of']
    df = cfg['df']
    num_splits = cfg['num_freq_splits']

    # Prepare Time Domain Wavelets
    wavelets_time = _normalize_wavelet_array(wv).astype(np.float32)
    if wavelets_time.shape[0] == 1:
        wavelets_time = np.repeat(wavelets_time, nshots, axis=0)

    # FFT
    freq_step_fft = 1.0 / (nt * dt)
    freqs = of + np.arange(nf) * df
    freq_idx = np.rint(freqs / freq_step_fft).astype(int)
    
    if freq_idx.min() < 0 or freq_idx.max() >= nt:
        raise ValueError("Requested frequency range exceeds wavelet Nyquist.")

    spec = np.fft.fft(wavelets_time, axis=1).astype(np.complex64)
    spec_full_band = spec[:, freq_idx] 

    # Split into Bands
    dfs = []
    n_band_samples = nf // num_splits
    
    for i in range(num_splits):
        idx_start = i * n_band_samples
        idx_end = nf if i == num_splits - 1 else idx_start + n_band_samples
        
        current_nf = idx_end - idx_start
        current_of = of + (idx_start * df)
        w_slice = spec_full_band[:, idx_start:idx_end]
        
        band_sep_vectors = []
        for shot_idx in range(nshots):
            wav = SepVector.getSepVector(
                ns=[current_nf], ds=[df], os=[current_of], storage="dataComplex"
            )
            wav[:] = w_slice[shot_idx]
            band_sep_vectors.append(wav)
            
        dfs.append(pd.DataFrame({
            "uniqueshots": np.arange(nshots, dtype=np.int32),
            "data": band_sep_vectors,
            "freq_band_id": i 
        }))
        
    return pd.concat(dfs, ignore_index=True).sort_values(['freq_band_id', 'uniqueshots']).reset_index(drop=True)


def load_geometry(cfg: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], int]:
    shot_col = cfg['shot_col']
    cols = [shot_col, cfg['sx_col'], cfg['sz_col'], cfg['rx_col'], cfg['rz_col']]
    
    use_sy = cfg['sy_col'] not in cols
    use_ry = cfg['ry_col'] not in cols
    if not use_sy: cols.append(cfg['sy_col'])
    if not use_ry: cols.append(cfg['ry_col'])

    geom_df = ds.dataset(cfg['geometry']).to_table(columns=cols).to_pandas()
    geom_df = geom_df.sort_values(shot_col).reset_index(drop=True)

    shot_ids = geom_df[shot_col].to_numpy()
    uniq_ids = np.unique(shot_ids)
    id_map = {sid: i for i, sid in enumerate(uniq_ids)}
    geom_df["shot_idx"] = np.array([id_map[sid] for sid in shot_ids], dtype=np.int32)

    src_df = geom_df.drop_duplicates(subset=["shot_idx"], keep="first").sort_values("shot_idx")

    sy_src = src_df[cfg['sy_col']].to_numpy(np.float32) if not use_sy else np.full(len(src_df), cfg['default_y'], np.float32)
    ry_rec = geom_df[cfg['ry_col']].to_numpy(np.float32) if not use_ry else np.full(len(geom_df), cfg['default_y'], np.float32)

    geometry = {
        "sx": src_df[cfg['sx_col']].to_numpy(np.float32),
        "sy": sy_src,
        "sz": src_df[cfg['sz_col']].to_numpy(np.float32),
        "s_ids": src_df["shot_idx"].to_numpy(np.int32),
        "rx": geom_df[cfg['rx_col']].to_numpy(np.float32),
        "ry": ry_rec,
        "rz": geom_df[cfg['rz_col']].to_numpy(np.float32),
        "r_ids": geom_df["shot_idx"].to_numpy(np.int32),
    }
    return geometry, len(src_df)


def build_par(cfg: Dict[str, Any]):
    keys = [
        "nref", "eps", "padx", "pady", "taperx", "tapery",
        "ref_look_ahead", "compress_error", "wflds_to_store", "wfld_path"
    ]
    return dict({k: cfg[k] for k in keys if k in cfg})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    print(f"--- Starting FWIX Modeling ---")
    print(f"Config: {args.config}")
    
    # Initialize Dask Client using your util
    dask_client = dask_util.load('Par/client.dask')
    client = dask_client.getClient()
    
    # Load Components
    geometry, nshots = load_geometry(cfg)
    model = load_model_split(cfg)
    wavelet_df = build_wavelet_df_split(cfg, nshots)
    par = build_par(cfg)

    os.makedirs(os.path.dirname(cfg['output']) or '.', exist_ok=True)

    pipeline = DaskPipeline(
        [
            FWIXmodeling.FWIXmodeling(
                model,
                wavelet_df,
                par,
                geometry,
                partition_size=cfg['partition_size'],
                shots_per_gpu=cfg['shots_per_gpu'],
                gpu_stream_batches=tuple(cfg['gpu_batches']),
            ),
            ComplexToFloat(col="data"),
            PyArrowWriter(cfg['output']),
        ]
    ).execute()

    print(f"Modeling completed successfully.")
    print(f"Output: {cfg['output']}")

if __name__ == "__main__":
    main()