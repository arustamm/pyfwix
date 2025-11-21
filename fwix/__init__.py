# src/fwix/__init__.py

# 1. Dynamic Versioning (best practice)
# Gets the version defined in pyproject.toml installed metadata
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("fwix")
except (ImportError, PackageNotFoundError):
    # Fallback if package is not installed yet (e.g. local dev)
    __version__ = "0.0.0"

# 2. Import Submodules
# We import the subpackages so users can do 'fwix.operator.run_kernel()'
# If these fail, we WANT them to crash immediately so Dask reports the error.
from . import CudaOperator
from . import CudaWEM

# 4. Define Public API
__all__ = ["CudaOperator", "CudaWEM"]