import pandas as pd
import numpy as np
import dask
import pyVector as Vector
import pyOperator as Operator
from pyProblem import Problem  # Assuming base class is in pyProblem.py
from pyParquetVector import pyParquetVector  # Assuming this is your Dask-compatible vector
import SepVector
import Hypercube
import gc

import pandas as pd
import numpy as np
import dask
import pyVector as Vector
import SepVector
import Hypercube
import genericIO
import gc
import pyCudaWEM as Prop 

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

def _obj_grad(
    df: pd.DataFrame,
    model_super: Vector.vector, # [Velocity, Density] (Physical units)
    wavelet: SepVector.Vector,
    par_dict
) -> pd.DataFrame:
    
    # slowest axis is the frequency axis
    ns = model_super.vecs[0].shape[0].n
    ds = model_super.vecs[0].shape[0].d
    os = model_super.vecs[0].shape[0].o

    # 1. Load Data
    data = df_to_sepvector(df, ns, ds, os)
    
    # 2. Unpack Model
    vel = model_super.vecs[0] 
    den = model_super.vecs[1]

    # 3. Extract Geometry
    geometry = {
        "sx": df['sx'].values.astype(np.float32),
        "sy": df['sy'].values.astype(np.float32),
        "sz": df['sz'].values.astype(np.float32),
        "rx": df['rx'].values.astype(np.float32),  
        "ry": df['ry'].values.astype(np.float32),
        "rz": df['rz'].values.astype(np.float32),
        "s_ids": df['uniquerecid'].values.astype(np.int32),
        "r_ids": df['r_ids'].values.astype(np.int32) if 'r_ids' in df else np.zeros(len(df), dtype=np.int32)
    }

    # 4. Instantiate Operators (Using Physical Models)
    parObj = genericIO.pythonParams(par_dict)
    nl_op = Prop.StreamingPropagator([vel, den], data, wavelet, parObj, geometry)
    g_op = Prop.StreamingExtendedBorn([vel, den], data, [vel, den], nl_op)

    res = data.clone()
    grad = model_super.clone().zero()
    
    # 5. Forward & Residual
    nl_op.forward(False, [vel, den], res)
    res.scaleAdd(data, 1.0, -1.0)
    
    # 6. Objective
    norm_sq = res.dot(res)
    
    # 7. Adjoint (Physical Gradient)
    g_op.adjoint(False, grad, res)

    del nl_op, g_op, res, data
    gc.collect()
    
    return pd.DataFrame({"norm_sq": [norm_sq], "grad": [grad]})

def _obj(
    df: pd.DataFrame,
    model: Vector.vector,
    wavelet: SepVector.Vector,
    par
) -> pd.DataFrame:
    """
    Computes objective for a single data partition.
    
    Args:
        df: A pandas DataFrame representing one partition of the data.
        model: The current model vector (fits in memory).
        nlop: The non-linear operator (or a function to create it).
    
    Returns:
        A single-row pandas DataFrame with 'norm_sq'.
    """
    
    # 1. Convert pandas partition to a pyVector
    data = df_to_sepvector(df)
    
    # Create geometry
    geometry = {
        "sx": df['sx'].values,
        "sy": df['sy'].values,
        "sz": df['sz'].values,
        "rx": df['rx'].values,  
        "ry": df['ry'].values,
        "rz": df['rz'].values,
    }

    # 2. Instantiate operators for this partition
    nl_op = Prop.StreamingPropagator(model, data, wavelet, par, geometry)

    # Example for a standard L2 problem: r = f(m) - d
    res = data.clone()
    grad = model.clone().zero()
    
    # 3. Forward: r = f(m) - d
    nl_op.forward(False, model, res)  # res = f(m)
    res.scaleAdd(data, 1.0, -1.0)   # res = f(m) - d
    
    # 4. Objective: J = ||r||^2
    norm_sq = res.dot(res)

    gc.collect()
    
    # 6. Return as a single-row DataFrame
    # The 'grad' column contains the full gradient vector object for this partition.
    return pd.DataFrame({"norm_sq": [norm_sq]})