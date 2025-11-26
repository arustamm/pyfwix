import sys
import os
import site
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Configure fwix environment dependencies")
    # We ask for the BASE folder of sep-io to be safe
    parser.add_argument("--sep-home", required=True, help="Root path of SEP-IO (e.g. ~/.local/sep-io)")
    args = parser.parse_args()

    sep_home = os.path.abspath(os.path.expanduser(args.sep_home))
    
    # 1. Define critical paths
    # We add BOTH lib and lib64 to be safe
    path_python = os.path.join(sep_home, "lib", "python3.11")
    path_lib = os.path.join(sep_home, "lib")
    path_lib64 = os.path.join(sep_home, "lib64")

    # --- TASK A: Setup Python Imports (.pth) ---
    site_packages = site.getsitepackages()[0]
    pth_file = os.path.join(site_packages, "fwix_deps.pth")
    
    print(f"[1/2] Linking Python path: {path_python}")
    try:
        with open(pth_file, "w") as f:
            f.write(path_python + "\n")
    except PermissionError:
        print("Error: Permission denied writing to site-packages.")
        sys.exit(1)

    # --- TASK B: Setup LD_LIBRARY_PATH (Conda Activation) ---
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        print(f"[2/2] Configuring LD_LIBRARY_PATH for Conda...")
        
        act_dir = Path(conda_prefix) / "etc" / "conda" / "activate.d"
        deact_dir = Path(conda_prefix) / "etc" / "conda" / "deactivate.d"
        act_dir.mkdir(parents=True, exist_ok=True)
        deact_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create Activation Script
        act_script = act_dir / "fwix_vars.sh"
        with open(act_script, "w") as f:
            f.write("#!/bin/sh\n")
            f.write("export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH\n")
            # Add BOTH lib and lib64
            f.write(f"export LD_LIBRARY_PATH=\"{path_lib}:{path_lib64}:$LD_LIBRARY_PATH\"\n")
        
        # 2. Create Deactivation Script
        deact_script = deact_dir / "fwix_vars.sh"
        with open(deact_script, "w") as f:
            f.write("#!/bin/sh\n")
            f.write("export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH\n")
            f.write("unset OLD_LD_LIBRARY_PATH\n")

        print("Done. Please run: conda deactivate && conda activate fwix")
    else:
        print("Warning: Not in a Conda env. Please set LD_LIBRARY_PATH manually.")

if __name__ == "__main__":
    main()