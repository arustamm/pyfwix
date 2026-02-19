import pandas as pd
import numpy as np
import dask
import pyVector as Vec
import pyOperator as Operator
from pyProblem import Problem  # Assuming base class is in pyProblem.py
import SepVector
import Hypercube
import gc
import dask.array as da
from pyVector import superVector
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import dask
import pyVector as Vector
from pyZarrVector import ZarrVector
import SepVector
import Hypercube
import genericIO
import gc
from scipy.ndimage import gaussian_filter

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
    r_ids = df[geom_mapping['id']].values.astype(np.int32)

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
        if traces.shape != vec.shape:
            raise ValueError(f"Data shape mismatch: expected {vec.shape}, got {traces.shape}")
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

def slice_sepvector(vec: SepVector, slices):
    arr = vec[slices] 
    ax = vec.getHyper().axes
    ns = [ax[i].n for i in range(len(ax))]
    os = [ax[i].o for i in range(len(ax))]
    ds = [ax[i].d for i in range(len(ax))]

    new_ns = list(reversed(arr.shape))
    new_ds = list(ds)
    new_os = list(os)
    
    ndim = len(vec.shape)
    for i, sl in enumerate(slices):
        start = sl.start if isinstance(sl, slice) else sl
        if start is None: start = 0
        idx = ndim - 1 - i
        new_os[idx] = os[idx] + start * ds[idx]

    vec = SepVector.getSepVector(ns=new_ns, os=new_os, ds=new_ds, storage='dataComplex')
    vec[:] = arr[:]
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

def get_slices(geometry, slowness, padx, pady):
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
    
    axes = slowness.getHyper().axes
    ox, oy, of, oz = [ax.o for ax in axes]
    dx, dy, df, dz = [ax.d for ax in axes]
    nx, ny, nf, nz = [ax.n for ax in axes]

    limitx = ox + (nx-1) * dx
    limity = oy + (ny-1) * dy

    # Add Padding (Aperture)
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

def prepare_extended_model(model, nf, of, df, pad_z=0) -> ZarrVector:
    if of == 0:
        raise ValueError("Cannot run zero frequency!")
    
    axes = model.getHyper().axes
    nz, ny, nx = axes[0].n, axes[1].n, axes[2].n
    oz, oy, ox = axes[0].o, axes[1].o, axes[2].o
    dz, dy, dx = axes[0].d, axes[1].d, axes[2].d
    n_pad = int(round(pad_z / dz))
    
    # 2. Lazy pad
    model_padded = np.pad(
        model[:],
        pad_width=((0, 0), (0, 0), (n_pad, 0)),
        mode='edge',
        # constant_values=1.5
    )
    
    # 3. (nz, ny, nx+n_pad) -> (nz+n_pad, ny, nx)
    transposed = np.transpose(model_padded, (2, 0, 1))
    
    # 4. Lazy compute 1/v^2
    inv_squared = 1.0 / (transposed ** 2)
    inv_squared = inv_squared.astype(np.complex64)
    
    # 5. Add frequency dimension and broadcast
    # (nz+n_pad, ny, nx) -> (nz+n_pad, ny, nx, 1) -> (nz+n_pad, ny, nx, nf)
    extended = np.broadcast_to(
        inv_squared[:, np.newaxis, :, :],
        shape=(nz + n_pad, nf, nx, ny)
    )
    
    ext_model = SepVector.getSepVector(
        ns = [ny, nx, nf, nz + n_pad],
        os = [oy, ox, of, 0.0],
        ds = [dy, dx, df, dz], storage='dataComplex'
    )
    ext_model[:] = extended[:]

    return ext_model

import concurrent.futures
import numpy as np
import SepVector
from pyVector import superVector
from scipy.ndimage import gaussian_filter

def create_split_model(slow_full: SepVector.vector, den_full: SepVector.vector, n_splits: int):
    """
    Splits monolithic 4D Slowness and Density vectors into a 
    SuperVector of SuperVectors based on frequency bands.
    Parallelized version.
    """
    # 1. Inspect Geometry
    hyp = slow_full.getHyper()
    axes = hyp.axes
    
    nx, ny, nf, nz = axes[0].n, axes[1].n, axes[2].n, axes[3].n
    dx, dy, df, dz = axes[0].d, axes[1].d, axes[2].d, axes[3].d
    ox, oy, of, oz = axes[0].o, axes[1].o, axes[2].o, axes[3].o

    n_band = nf // n_splits
    
    # Get Numpy Views (Shape: nz, nf, ny, nx)
    slow_arr = slow_full.getNdArray()
    den_arr = den_full.getNdArray()

    print(f"Splitting model with {nf} freqs into {n_splits} bands (Parallel)...")

    # --- Worker Function ---
    def _split_worker(i):
        # A. Calculate Indices
        idx_start = i * n_band
        if i == n_splits - 1:
            idx_end = nf
        else:
            idx_end = idx_start + n_band
            
        current_nf = idx_end - idx_start
        current_of = of + (idx_start * df)
        
        # B. Define Hypercube for this Band
        band_ns = [nx, ny, current_nf, nz]
        band_ds = [dx, dy, df, dz]
        band_os = [ox, oy, current_of, oz]
        
        # C. Create SepVectors
        s_vec = SepVector.getSepVector(
            ns=band_ns, ds=band_ds, os=band_os, 
            storage='dataComplex'
        )
        d_vec = SepVector.getSepVector(
            ns=band_ns, ds=band_ds, os=band_os, 
            storage='dataComplex'
        )
        
        # D. Fill Data (Copy from Monolithic)
        # Note: Concurrent reads from slow_arr/den_arr are safe
        s_vec[:] = slow_arr[:, idx_start:idx_end, :, :]
        d_vec[:] = den_arr[:, idx_start:idx_end, :, :]
        
        # E. Return the SuperVector for this band
        return superVector(s_vec, d_vec)

    # --- Execution ---
    # Use max_workers=n_splits to do it all at once if memory allows
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_splits) as executor:
        # Submit all tasks. We map indices 0..n_splits-1
        # We store the index 'i' in the map so we know where to put the result
        futures = {executor.submit(_split_worker, i): i for i in range(n_splits)}
        
        # Prepare the list with empty slots to ensure correct ordering
        band_supervectors = [None] * n_splits
        
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                band_supervectors[idx] = result
            except Exception as e:
                print(f"Split {idx} generated an exception: {e}")
                raise e

    # F. Create Master SuperVector (unpacked in correct order)
    final_model = superVector(*band_supervectors)
    return final_model


def create_grad_mask(model: SepVector.vector, zeros: list[float], sigma=3.0):
    """
    Applies Gaussian taper to water layer in parallel across model splits.
    Input 'model' is expected to be the SuperVector of splits created above.
    """
    mask = model.clone()
    mask.set(1.)
    
    # Helper to calculate nzeros once (assuming all splits have same dz)
    # Peek at the first component of the first split
    first_split = mask.vecs[0] # This is a SuperVector(Slow, Den)
    first_comp = first_split.vecs[0] # This is the Slowness SepVector
    dz = first_comp.getHyper().axes[-1].d # Axis 3 is Z (in Python [X,Y,F,Z])
    ztop, zbottom = int(zeros[0] / dz), int(zeros[1] / dz)

    # --- Worker Function ---
    def _mask_worker(split_idx, split_sv):
        """
        Process one frequency band (SuperVector containing Slow and Den)
        """
        
        # split_sv is a SuperVector([Slow, Den])
        # vec[0] is Slowness, vec[1] is Density
        
        # Apply to Slowness
        slow = split_sv.vecs[0].getNdArray()
        slow[:ztop, ...] = 0.
        slow[-zbottom:, ...] = 0.
        slow[:] = gaussian_filter(slow, sigma=sigma, axes=0)
        
        # Apply to Density
        # den = split_sv.vecs[1].getNdArray()
        # den[:ztop, ...] = 0.
        # den[-zbottom:, ...] = 0.
        # den[:] = gaussian_filter(den, sigma=sigma, axes=0)
        
        return split_idx

    # --- Execution ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(mask.vecs)) as executor:
        # mask.vecs contains the list of splits (SuperVectors)
        futures = []
        for i, split_sv in enumerate(mask.vecs):
            futures.append(executor.submit(_mask_worker, i, split_sv))
            
        for future in concurrent.futures.as_completed(futures):
            # Just check for exceptions, modification happens in-place
            try:
                _ = future.result()
            except Exception as e:
                print(f"Masking failed on split: {e}")
                raise e
    
    return mask, ztop, zbottom


import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.signal import windows

def compute_structural_dips(image_tlg, dx, dy, dz, sigma=1.0):
    """
    Computes structural dips z_x and z_y from a Time-Lag Gather.
    
    Parameters:
    -----------
    image_tlg : ndarray
        3D volume (nz, ny, nx) - typically zero-lag image
    dx, dy, dz : float
        Grid spacing in physical units (e.g., meters or km).
    sigma : float
        Smoothing factor for stability. standard=1.0 pixel.
        
    Returns:
    --------
    zx, zy : ndarray
        3D volumes of dips. Shape: (nz, ny, nx).
    """
    
    # 1. Smooth Image to Stabilize Derivatives
    img_smooth = gaussian_filter(image_tlg, sigma=sigma)
    
    # 2. Compute Gradients
    # np.gradient returns [dI/dim0, dI/dim1, dI/dim2]
    # Assuming input shape is (Z, Y, X)
    g_z, g_y, g_x = np.gradient(img_smooth, dz, dy, dx, edge_order=2)
    
    # 3. Calculate Dips (zx, zy)
    # Formula: zx = - (dI/dx) / (dI/dz)
    
    # Add stabilization term to denominator to avoid division by zero
    epsilon = 1e-2 * (np.max(np.abs(g_z)) + 1e-10)
    
    zx = -g_x / (g_z + epsilon)
    zy = -g_y / (g_z + epsilon)
    
    return zx, zy

def get_slope_from_eq26(theta_rad, s_z, zx_z, zy_z, azimuth=0.0):
    """
    Compute tau-slope from angle using Sava & Fomel eq. 26.
    
    Parameters:
    -----------
    theta_rad : float or ndarray
        Opening angle in radians
    s_z : ndarray
        Slowness profile s(z). Shape: (nz,)
    zx_z, zy_z : ndarray
        Structural dip profiles. Shape: (nz,)
    azimuth : float
        Azimuth angle in radians (for 3D). Default 0 (inline direction).
        
    Returns:
    --------
    slope : ndarray
        d(tau)/dz in physical units (s/m)
    """
    # Dip correction factor: sqrt(1 + zx^2 + zy^2)
    struct_term = np.sqrt(1 + zx_z**2 + zy_z**2)
    
    # For 2D or inline direction (azimuth = 0):
    # slope = s(z) * cos(theta) / sqrt(1 + zx^2 + zy^2)
    # This is the inverse of eq. 26 solved for p_tau
    
    # CORRECTED: The slope should be:
    # d(tau)/dz = s(z) * cos(theta) / struct_term
    slope = (s_z * np.cos(theta_rad)) / (struct_term + 1e-10)
    
    return slope

def tau_to_adcig_local(gather_1d, s_z, zx_z, zy_z, dz, dtau, angles_deg, 
                 window_z=20, azimuth=0.0):
    """
    Converts Time-Lag gather to Angle Gather for a SINGLE (x,y) location.
    
    Parameters:
    -----------
    gather_1d : ndarray
        Time-Lag gather at this location. Shape: (nz, ntau)
    s_z : ndarray
        Slowness profile s(z) at this location. Shape: (nz,)
    zx_z, zy_z : ndarray
        Structural dip profiles z_x(z) and z_y(z). Shape: (nz,)
    dz, dtau : float
        Grid spacing in depth and tau directions (same units as slowness)
    angles_deg : ndarray
        Target opening angles in degrees.
    window_z : int
        Half-width of the vertical stacking window (in samples)
    azimuth : float
        Azimuth angle in degrees (for 3D)
        
    Returns:
    --------
    angle_gather : ndarray
        The angle gather for this point. Shape: (nz, n_angles)
    """
    nz, ntau = gather_1d.shape
    n_angles = len(angles_deg)
    angles_rad = np.radians(angles_deg)
    azimuth_rad = np.radians(azimuth)
    
    # Output container
    angle_gather = np.zeros((nz, n_angles), dtype=np.float32)
    
    # Tau center index (zero-lag position)
    tau_center_idx = ntau // 2
    
    # Define the local window for stacking
    z_offsets = np.arange(-window_z, window_z + 1)
    n_stack = len(z_offsets)
    
    # Loop through desired angles
    for i_ang, theta in enumerate(angles_rad):
        
        # 1. Calculate the slope profile for this angle
        # Units: s/m (physical slope d(tau)/dz)
        slope_profile_phys = get_slope_from_eq26(
            theta, s_z, zx_z, zy_z, azimuth_rad
        )
        
        # Convert to index units: (tau_samples / z_samples)
        slope_profile_idx = slope_profile_phys * (dz / dtau)
        
        # 2. Stack along the trajectory
        for iz in range(nz):
            
            amp_sum = 0.0
            count = 0
            
            for dz_offset in z_offsets:
                iz_neighbor = iz + dz_offset
                
                # Boundary check in z
                if iz_neighbor < 0 or iz_neighbor >= nz:
                    continue
                
                # Compute tau shift based on slope at current depth iz
                # The trajectory is: tau(z') = tau_0 + slope(z) * (z' - z)
                tau_shift = slope_profile_idx[iz] * dz_offset
                itau_float = tau_center_idx + tau_shift
                
                # Boundary check in tau
                if itau_float < 0 or itau_float >= ntau - 1:
                    continue
                
                # 3. Bilinear interpolation
                itau_floor = int(np.floor(itau_float))
                itau_ceil = itau_floor + 1
                weight = itau_float - itau_floor
                
                if itau_ceil < ntau:
                    amp = (1 - weight) * gather_1d[iz_neighbor, itau_floor] + \
                          weight * gather_1d[iz_neighbor, itau_ceil]
                else:
                    amp = gather_1d[iz_neighbor, itau_floor]
                
                amp_sum += amp
                count += 1
            
            # Normalize by number of valid samples
            if count > 0:
                angle_gather[iz, i_ang] = amp_sum
        
    return angle_gather

def freq_to_timelag(freq_gather, dt, nt, fmin=3, fmax=25, taper_alpha=0.2, max_time_lag=None):
    """
    Convert frequency-domain gather to time-lag domain, optionally windowing 
    to a specific maximum lag.
    
    Parameters:
    -----------
    freq_gather : ndarray
        Frequency domain gather. Shape: (nz, nfreq, ...) 
    dt : float
        Time sampling interval
    nt : int
        Original number of time samples (determines df)
    max_time_lag : float or None
        If set, truncates the output to keep lags within [-max_time_lag, +max_time_lag].
        This significantly reduces memory for the subsequent angle conversion.
    
    Returns:
    --------
    timelag_gather : ndarray
        Time-lag domain gather. 
        Shape: (nz, nt) if max_time_lag is None
        Shape: (nz, 2*n_lag + 1) if max_time_lag is set
    """
    nz = freq_gather.shape[0]
    nf = freq_gather.shape[1]

    # Calculate frequency indices
    ifmin = int(round(fmin * dt * nt))
    ifmax = ifmin + nf
    
    # Create Tukey taper
    taper = windows.tukey(nf, alpha=taper_alpha)
    
    # Initialize frequency-domain array (full spectrum size based on nt)
    f_grad = np.zeros((nz, nt), dtype=np.complex64)
    
    # Fill positive frequencies
    f_grad[:, ifmin:ifmax] = freq_gather * taper[None, :]
    
    # Fill negative frequencies (Hermitian symmetry)
    f_grad[:, -ifmax:-ifmin] = np.conj(np.flip(
        freq_gather * taper[None, :], axis=1
    ))
    
    # Transform to time domain
    t_grad = np.fft.ifft(f_grad, axis=1).real
    
    # Shift zero-lag to center
    t_grad = np.fft.fftshift(t_grad, axes=1)
    
    # --- NEW: Apply Time Lag Windowing ---
    if max_time_lag is not None:
        # Calculate number of samples to keep on each side of zero
        n_lag_samples = int(np.floor(max_time_lag / dt))
        
        center_idx = nt // 2
        start_idx = center_idx - n_lag_samples
        end_idx = center_idx + n_lag_samples + 1
        
        # Ensure we don't go out of bounds
        start_idx = max(0, start_idx)
        end_idx = min(nt, end_idx)
        
        # Slice the array
        t_grad = t_grad[:, start_idx:end_idx]
        
    return t_grad

def process_single_location_from_freq(args):
    """
    Worker function to process a single (ix, iy) location from frequency domain.
    Updated to unpack max_time_lag.
    """
    (ix, iy, freq_gather, s_z, zx_z, zy_z, 
     dt, nt, dz, dx, dy, angles_deg, window_z, azimuth,
     fmin, fmax, taper_alpha, max_time_lag) = args  # <--- Added max_time_lag to unpack
    
    # Step 1: Convert frequency domain to time-lag domain (with optional windowing)
    timelag_gather = freq_to_timelag(
        freq_gather, dt, nt, fmin, fmax, taper_alpha, max_time_lag
    )
    
    # Determine dtau from the time-lag gather
    # Note: dtau is just dt, but ntau might now be smaller due to windowing
    dtau = dt 
    
    # Step 2: Convert time-lag to angle domain
    # tau_to_adcig_local automatically handles the new shape because it 
    # calculates the center based on gather_1d.shape[1] // 2
    angle_gather = tau_to_adcig_local(
        timelag_gather, s_z, zx_z, zy_z, 
        dz, dtau, angles_deg, 
        window_z=window_z, azimuth=azimuth
    )
    
    return (ix, iy, angle_gather)

def freq_to_angle_volume(freq_volume, slowness_volume, zx_volume, zy_volume,
                        dt, nt, dz, dx, dy, angles_deg,
                        fmin, fmax, taper_alpha=0.2,
                        window_z=20, azimuth=0.0, 
                        max_time_lag=None,   # <--- New Parameter
                        n_processes=None, chunk_size=50, verbose=True):
    """
    Convert frequency-domain volume to angle gathers in parallel.
    
    Parameters:
    -----------
    ... (standard args) ...
    max_time_lag : float or None
        Maximum time lag (in seconds) to keep before angle conversion.
        Events outside [-max_time_lag, +max_time_lag] are discarded.
        Set this to reduce memory usage and compute time.
    """
    nz, nfreq, ny, nx = freq_volume.shape
    n_angles = len(angles_deg)
    
    if verbose:
        print(f"Input frequency volume shape: {freq_volume.shape}")
        if max_time_lag:
            print(f"Applying max time lag: +/- {max_time_lag} s")
        else:
            print("Keep full time axis (no lag truncation)")
            
    # Initialize output volume
    angle_volume = np.zeros((nz, n_angles, ny, nx), dtype=np.float32)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    
    # Prepare arguments for all spatial locations
    tasks = []
    for iy in range(ny):
        for ix in range(nx):
            freq_gather = freq_volume[:, :, iy, ix]
            s_z = slowness_volume[:, iy, ix]
            zx_z = zx_volume[:, iy, ix]
            zy_z = zy_volume[:, iy, ix]
            
            # Pack arguments (added max_time_lag at the end)
            tasks.append((ix, iy, freq_gather, s_z, zx_z, zy_z,
                         dt, nt, dz, dx, dy, angles_deg, window_z, azimuth,
                         fmin, fmax, taper_alpha, max_time_lag))
    
    # Process in parallel
    total_tasks = len(tasks)
    completed = 0
    
    with Pool(processes=n_processes) as pool:
        for ix, iy, angle_gather in pool.imap_unordered(
            process_single_location_from_freq, 
            tasks, 
            chunksize=chunk_size
        ):
            angle_volume[:, :, iy, ix] = angle_gather
            
            completed += 1
            if verbose and completed % 100 == 0:
                percent = 100 * completed / total_tasks
                print(f"Progress: {completed}/{total_tasks} ({percent:.1f}%)")
    
    if verbose:
        print("Conversion complete!")
    
    return angle_volume