#!/bin/bash

# An improved script that builds and installs python-solver from source
# using its own CMake build system.
# This should be run AFTER build_sep_io.sh has completed.

# --- Configuration ---
REPO="https://github.com/arustamm/pythonSolver.git"
BRANCH="rustam/constrained"
# This MUST match the installation directory from the build_sep_io.sh script.
INSTALL_DIR="${HOME}/.local/sep-io-fwix/lib/python3.11"
# Temporary directory to clone and build the solver.
SOLVER_SRC_DIR="/tmp/python-solver-src"

# Exit immediately if a command fails
set -e

echo "--- Updating Python Solver (Direct Copy Method) ---"
echo "Target installation directory: ${INSTALL_DIR}"

# --- 1. Verify the sep-io installation exists ---
if [ ! -d "$INSTALL_DIR" ]; then
    echo "ERROR: sep-io installation not found at ${INSTALL_DIR}"
    echo "Please run the ./build_sep_io.sh script first."
    exit 1
fi

# --- 2. Clone the specific python-solver branch ---
echo "Cloning the '$BRANCH' branch of python-solver..."
rm -rf "$SOLVER_SRC_DIR"
git clone --branch "$BRANCH" "$REPO" "$SOLVER_SRC_DIR"

# --- 3. Identify Source and Destination Directories ---
# The Python code inside the solver repository
SOLVER_PYTHON_SOURCE="${SOLVER_SRC_DIR}/GenericSolver/python"

echo "Source directory: ${SOLVER_PYTHON_SOURCE}"
echo "Destination directory: ${INSTALL_DIR}"

# --- 4. Copy all .py files ---
# Find all .py files in the source and copy them to the destination, overwriting.
echo "Copying new python-solver .py files and overwriting old ones..."
find "$SOLVER_PYTHON_SOURCE" -name "*.py" -exec cp -v {} "$INSTALL_DIR" \;

echo "--- Python Solver update complete! ---"