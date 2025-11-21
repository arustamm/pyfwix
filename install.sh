#!/bin/bash
set -e  # Exit on error

# 1. Create the Conda Environment
echo "Creating Conda environment 'fwix'..."
conda env create -f environment.yaml || conda env update -f environment.yaml --prune

# 2. Activate the environment for the rest of the script
# (This trick allows activation inside a script)
eval "$(conda shell.bash hook)"
conda activate fwix

echo "Environment active. Installing custom libraries..."

echo "---------------------------------------"
echo "Installation Complete!"
echo "Run: conda activate fwix"

echo "---------------------------------------"
echo "Installing sep-io libraries..."
./build_sep_io.sh
echo "sep-io libraries installed."

echo "---------------------------------------"
echo "Installing python-solver..."
./build_python_solver.sh
echo "python-solver installed."

echo "---------------------------------------"
echo "Installing pyfwix..."
pip install .
echo "pyfwix installed."

echo "---------------------------------------"
echo "Testing the installation..."
python -c "import genericIO; print('genericIO imported successfully')"
python -c "import SepVector; print('SepVector imported successfully')"
python -c "import pyVector; print('pyVector imported successfully')"
python -c "from fwix import CudaWEM; print('CudaWEM imported successfully')"
python -c "from fwix import CudaOperator; print('CudaOperator imported successfully')"
echo "All tests passed. Installation successful!"