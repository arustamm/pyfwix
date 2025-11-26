#!/bin/bash

# An "ultimate clean" script that forcefully removes submodules before
# updating and dynamically patches all CMakeLists.txt files.

# --- Configuration ---
INSTALL_DIR="${HOME}/.local/sep-io-fwix"
SOURCE_DIR="/tmp/sep-io-build-src"
PYBIND11_VERSION="v2.13.6"

# Exit immediately if a command fails
set -e

echo "--- Building SEP-IO (Ultimate Clean Mode) ---"
echo "Installation destination: ${INSTALL_DIR}"

# --- 1. Get the Source Code ---
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Cloning repository..."
    git clone http://zapad.stanford.edu/SEP-external/sep-iolibs.git "$SOURCE_DIR"
fi

cd "$SOURCE_DIR"
echo "Current working directory: $(pwd)"
if [[ "$(pwd)" != "$SOURCE_DIR" ]]; then
    echo "FATAL: Failed to change into the source directory. Aborting."
    exit 1
fi

# --- 2. Force Clean and Apply Patches ---
echo "--- Starting Aggressive Git Clean and Patch Process ---"

# Step 2a: AGGRESSIVE SUBMODULE CLEAN
echo "[Git Fix] Forcefully removing all submodule directories to guarantee a clean slate..."
# This command finds all submodule paths and removes them.
# The '|| true' prevents an error if there are no submodules found.
git submodule foreach --recursive 'rm -rf "$toplevel/$path"' || true
# Unregister the submodules from the .git/config file
git submodule deinit --all --force

# Step 2b: Re-initialize all submodules from scratch
echo "[Git Fix] Re-initializing all submodules from remote..."
git submodule update --init --recursive

# Step 2c: Checkout 'rustam-dev' branch in the desired submodule
echo "[Patch] Checking out rustam-dev branch in external/genericIO..."
git -C external/genericIO checkout --force rustam-dev
# ADD THIS: Stage the submodule changes (like Docker does)
echo "[Patch] Updating submodules *within* genericIO to match the rustam-dev branch..."
git -C external/genericIO submodule update --init --recursive

# Step 2d: Update the pybind11 submodule to the desired version
PYBIND11_PATH="external/genericIO/external/buffers/external/hypercube/external/pybind11"
echo "[Patch] Updating pybind11 submodule to '$PYBIND11_VERSION'..."
git -C "$PYBIND11_PATH" fetch --tags
git -C "$PYBIND11_PATH" checkout --force $PYBIND11_VERSION

# Step 2e: DYNAMICALLY FIND AND FIX ALL CMakeLists.txt files
echo "[Patch] Finding and fixing all CMakeLists.txt files to version 3.15..."
# This command finds every CMakeLists.txt and runs sed to patch it.
find . -name "CMakeLists.txt" -print0 | while IFS= read -r -d '' cmake_file; do
    echo "  -> Patching ${cmake_file}"
    sed -i.bak 's/cmake_minimum_required(VERSION [0-9]\.[0-9].*)/cmake_minimum_required(VERSION 3.15)/' "${cmake_file}"
done

echo "--- Patching Process Complete ---"

# --- 3. Configure, Build, and Install ---
echo "Configuring with CMake..."
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DBUILD_TEST=OFF \
    -DBUILD_SEP=ON \
    -DBUILD_SEGYIO=OFF \
    -DBUILD_GCP=OFF \
    -DBUILD_UTIL=OFF \
    -DBENCHMARK_DOWNLOAD_DEPENDENCIES=OFF \
    -DBENCHMARK_ENABLE_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_PYTHON=ON \
    -DBUILD_TEST=OFF \
    -DCMAKE_CXX_FLAGS="-O3 -ffast-math -DNDEBUG -DBOOST_DISABLE_ASSERTS -funroll-loops"

echo "Building with CMake..."
cmake --build build -j$(nproc)

echo "Installing to local directory..."
cmake --install build

echo "--- SEP-IO installation complete in ${INSTALL_DIR} ---"