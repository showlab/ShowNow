#!/bin/bash
# Portable torchrun launcher for the ShowNow training stages (FSDP, 8 GPUs per node).
#   single node:  scripts/train.sh <python.module> [args]
#   multi node:   NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<head ip> scripts/train.sh <python.module> [args]   (on every node)
#   Slurm:        srun --ntasks-per-node=1 scripts/train.sh ...   (NNODES / NODE_RANK / MASTER_ADDR from the environment)
# Environment: SHOWNOW_N_ACTIONS=9 (8 actions) and the data roots, see README.md.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# compiled flex attention for the block-causal masks; without it flex falls back to dense attention and OOMs
export SOLARWM_COMPILE_FLEX=1
NPROC=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
MODULE=$1; shift
if [ "$NNODES" = 1 ]; then
  exec torchrun --standalone --nproc-per-node "$NPROC" -m "$MODULE" "$@"
fi
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
: "${MASTER_ADDR:?set MASTER_ADDR to the head node address}"
exec torchrun --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "$NPROC" --rdzv-backend c10d \
  --rdzv-id "${RDZV_ID:-shownow}" --rdzv-endpoint "$MASTER_ADDR:${MASTER_PORT:-29500}" -m "$MODULE" "$@"
