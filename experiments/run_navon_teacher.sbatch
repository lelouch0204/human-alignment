#!/usr/bin/env bash
#SBATCH --job-name=navon-teacher
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/navon_shapes/teacher-%j.out
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/navon-teacher-uv}"
uv run --locked --no-sync python experiments/navon_teacher.py --checkpoint "$1"
