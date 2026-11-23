#!/bin/bash
#SBATCH --job-name=navon-spatial
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/navon_human/slurm-%j.out
##SBATCH --account=nparikh44

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Same cluster defaults as run_navon.sbatch; environment overrides are supported.
NAVON_UV_ROOT="${NAVON_UV_ROOT:-/storage/ice1/4/5/nparikh44}"
export PATH="$NAVON_UV_ROOT/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NAVON_UV_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$NAVON_UV_ROOT/.local/share/uv/python}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$NAVON_UV_ROOT/.local/share/uv/tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-$NAVON_UV_ROOT/.local/bin}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

for v in base unalignet; do
 uv run --locked --no-sync python experiments/navon_spatial_readouts.py "$v"
done
