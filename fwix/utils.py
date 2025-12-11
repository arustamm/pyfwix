import pandas as pd
import numpy as np
import dask
import pyVector as Vec
import pyOperator as Operator
from pyProblem import Problem  # Assuming base class is in pyProblem.py
import SepVector
import Hypercube
import gc

import pandas as pd
import numpy as np
import dask
import pyVector as Vector
from pyZarrVector import ZarrVector
import SepVector
import Hypercube
import genericIO
import gc

from typing import Tuple, Any, Dict

def df_to_sepvector(df: pd.DataFrame, ns: int, ds: float, os: float) -> Vector.vector:
    """Converts a dataframe partition to a SepVector."""
    ntraces = df.shape[0]
    # Stack arrays from 'data' column into a 2D block (ns, ntraces)
    data = np.stack(np.asarray(df['data'].values))
    if data.dtype != np.complex64:
        raise ValueError("Data array must be of type complex64.")
    
    axis1 = Hypercube.axis(n=ns, d=ds, o=os)
    axis2 = Hypercube.axis(n=ntraces, d=1.0, o=0.0)

    hyper = Hypercube.hypercube(axes=[axis1, axis2])
    vec = SepVector.getSepVector(hyper, storage='dataComplex')
    vec.getNdArray()[:] = data
    return vec

def get_axis(wavelet_df: pd.DataFrame) -> dict:
    """
    Extracts time axis from the first wavelet in the batch.
    """
    first_wav = wavelet_df.iloc[0]['data']
    ax = first_wav.getHyper().axes[0]
    return ax

def create_geometry(df: pd.DataFrame, geom_mapping: Dict[str, str]) -> dict:
    # 2. Construct Geometry for Batch
    rx = df[geom_mapping['rx']].values.astype(np.float32)
    ry = df[geom_mapping['ry']].values.astype(np.float32)
    rz = df[geom_mapping['rz']].values.astype(np.float32)
    r_ids = df[geom_mapping['id']].values.astype(np.float32)

    df_shots_unique = df.drop_duplicates(subset=[geom_mapping['id']], keep='first')
    sx = df_shots_unique[geom_mapping['sx']].values.astype(np.float32)
    sy = df_shots_unique[geom_mapping['sy']].values.astype(np.float32)
    sz = df_shots_unique[geom_mapping['sz']].values.astype(np.float32)
    s_ids = df_shots_unique[geom_mapping['id']].values.astype(np.int32)

    # For internal Cuda propagator and Born operators
    geometry = {
        "sx": sx, "sy": sy, "sz": sz, "s_ids": s_ids,
        "rx": rx, "ry": ry, "rz": rz, "r_ids": r_ids
    }

    return geometry

def create_data(df_batch: pd.DataFrame, axis) -> SepVector.vector:
    """
    Converts a dataframe batch (with a 'data' column) into a SepVector.
    """
    ntraces = len(df_batch)
    
    # Define Hypercube: Time Axis x Trace Axis
    axis_tr = Hypercube.axis(n=ntraces, d=1.0, o=0.0)
    hyper = Hypercube.hypercube(axes=[axis, axis_tr])
    
    # Create Vector
    vec = SepVector.getSepVector(hyper, storage='dataComplex')
    if 'data' in df_batch.columns:
        traces = np.stack(np.asarray(df_batch['data'].values))
        vec[:] = traces

    return vec

def create_wavelet(wav_df_batch: pd.DataFrame, axis) -> SepVector.vector:
    """
    Creates a SepVector containing wavelets for the current batch of shots.
    """
    n_shots = len(wav_df_batch)
    axis_s = Hypercube.axis(n=n_shots, d=1.0, o=0.0)
    hyper = Hypercube.hypercube(axes=[axis, axis_s])
    vec = SepVector.getSepVector(hyper, storage='dataComplex')

    for i, wav in enumerate(wav_df_batch['data']):
        vec[i, :] = wav[:]
    
    return vec

def zarr_to_sepvector(zvec: ZarrVector, slices=None):
    arr, ns, os, ds = zvec.to_numpy(slices)
    vec = SepVector.getSepVector(ns=ns, os=os, ds=ds, storage='dataComplex')
    vec[:] = arr[:]
    return vec

def sepvector_to_zarr(svec: SepVector.vector, path, temp_dir='/tmp/',
                      remove_file=False, chunks=None, shards=None) -> ZarrVector:
    if svec.getStorageType() == 'dataComplex':
        dtype = np.complex64
    else:
        dtype = np.float32
    ax = svec.getHyper().axes
    nss = [ax[i].n for i in range(len(ax))]
    oss = [ax[i].o for i in range(len(ax))]
    dss = [ax[i].d for i in range(len(ax))]

    zvec = ZarrVector(ns_list=nss, 
                      os_list=oss, 
                      ds_list=dss, 
                      chunks=chunks,
                      shards=shards,
                      path=path,
                      temp_dir=temp_dir,
                      overwrite=True,
                      remove_file=remove_file,
                      dtype=dtype)
    zvec[:] = svec[:]
    return zvec

def get_slices(geometry, slowness, pad_x, pad_y):
    """
    Calculates the bounding box for shots/receivers and extracts 
    the local model window and shifted geometry.
    """
    # 1. Calculate Global Bounding Box
    # We look at both Source and Receiver coordinates
    minx = min(geometry['sx'].min(), geometry['rx'].min())
    maxx = max(geometry['sx'].max(), geometry['rx'].max())
    
    miny = min(geometry['sy'].min(), geometry['ry'].min())
    maxy = max(geometry['sy'].max(), geometry['ry'].max())

    # 2. Add Padding (Aperture)
    xlen = maxx - minx
    ylen = maxy - miny
    padx = xlen + pad_x
    pady = ylen + pad_y
    
    # Clip to model boundaries
    # Assuming origin is 0,0 for simplicity, use slowness.os to be precise
    ox, oy, of, oz = slowness.os # Assuming [Z, Y, X] order
    dx, dy, df, dz = slowness.ds
    nx, ny, nf, nz = slowness.ns
    
    limitx = ox + (nx-1) * dx
    limity = oy + (ny-1) * dy

    startx = max(ox, minx - padx)
    endx   = min(limitx, maxx + padx)
    
    starty = max(oy, miny - pady)
    endy   = min(limity, maxy + pady)

    # 3. Convert Physical Coords to Indices (Slices)
    # Start index: (Inclusive) - Round is usually safe for grid-aligned data
    ixstart = int(np.round((startx - ox) / dx))
    
    # End index: (Exclusive) - We must ADD 1 to include the sample at 'endx'
    # We also clamp to 'nx' to prevent index-out-of-bounds
    ixend   = min(nx, int(np.round((endx - ox) / dx)) + 1)
    
    iystart = int(np.round((starty - oy) / dy))
    iyend   = min(ny, int(np.round((endy - oy) / dy)) + 1)
    
    # Slices for Zarr (X, Y, F, Z)
    slices = (
        slice(0, nz),          # Full Z
        slice(0, nf),          # Full F
        slice(iystart, iyend), # Y Window
        slice(ixstart, ixend),  # X Window
    )
    
    return slices

def prepare_extended_model(model, nf, of, df, path, pad_z = 0, 
                           chunks=None, shards=None,
                           remove_file=False, temp_dir='/tmp/') -> ZarrVector:
    if of == 0:
        raise ValueError("Cannot run zero frequency!")
    axes = model.getHyper().axes
    nz, ny, nx = axes[0].n, axes[1].n, axes[2].n
    oz, oy, ox = axes[0].o, axes[1].o, axes[2].o
    dz, dy, dx = axes[0].d, axes[1].d, axes[2].d
    n_pad = int(round(pad_z / dz))
    model_data = np.pad(
            model[:], 
            pad_width=((0, 0), (0, 0), (n_pad, 0)), 
            mode='constant', 
            constant_values=1.5
        ).astype(np.complex64)
    
    model_ext = ZarrVector(ns_list=[ny, nx, nf, nz + n_pad], 
                            ds_list=[dy, dx, df, dz], 
                            os_list=[oy, ox, of, oz], 
                            dtype=np.complex64, 
                            chunks=chunks,
                            shards=shards,
                            path=path,
                            temp_dir=temp_dir,
                            remove_file=remove_file)
    transpose_model = 1. / np.transpose(model_data[:], (2, 0, 1))**2

    # write to
    for i in range(nf):
        model_ext[:, i, :, :] = transpose_model
    return model_ext
