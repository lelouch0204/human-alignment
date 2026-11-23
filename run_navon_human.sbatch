#!/bin/bash
#SBATCH --job-name=navon-human
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
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

# Prepare the pinned public human data with the command in the guide before submission.
run=(uv run --locked --no-sync python -m human_alignment.navon_shapes)
data="${NAVON_SHAPES_DATA:-outputs/navon_shapes/data}"
cache="${NAVON_SHAPES_CACHE:-outputs/navon_shapes/embeddings}"
out="${NAVON_SHAPES_OUT:-outputs/navon_shapes/run-${SLURM_JOB_ID:-local}}"
"${run[@]}" prepare --data "$data"
echo "Node: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
nvidia-smi || true
model_flags=()
if [[ "${NAVON_DOWNLOAD_MODELS:-0}" == "1" ]]; then
    model_flags+=(--download-models)
fi
for family in dino siglip; do
    for variant in base unalignet alignet; do
        "${run[@]}" embed --data "$data" --cache "$cache" \
            --family "$family" --variant "$variant" --models "${NAVON_MODELS:-models}" \
            --batch-size "${NAVON_HUMAN_BATCH_SIZE:-16}" "${model_flags[@]}"
    done
done
"${run[@]}" analyze --data "$data" --cache "$cache" --out "$out"
