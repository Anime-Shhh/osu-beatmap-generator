#!/bin/bash
# Multi-GPU training of one v3 stage. Crash-safe:
#  * Slurm sends USR1 10 min before the time limit -> the trap touches RUN/STOP ->
#    the trainer saves a full checkpoint and requeues this same job, which resumes.
#  * Periodic checkpoints every 20 min bound the loss from a hard crash/preemption.
#  * Resubmitting the identical command at any time resumes from RUN/last.pt.
# Usage: sbatch [--gres=gpu:N] train_stage.sbatch RUN_DIR STAGE [trainer args...]
#SBATCH --job-name=v3-train
#SBATCH --partition=gpu
#SBATCH --account=general
# Amarel "k" nodes (halk*, gpuk*) see a diverged, stale copy of /scratch and /home since 2026-11-24;
# jobs there die instantly (signal 53) or write where the login side never sees it.
#SBATCH --exclude=gpuk[001-018],halk[0001-0159]
#SBATCH --gres=gpu:4
#SBATCH --constraint=adalovelace|ampere
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/asj102/osu_project/logs/v3-train-%x-%j.log
source "$SLURM_SUBMIT_DIR/scripts/amarel/v3/common.sh"
RUN="${1:?run dir}"; STAGE="${2:?stage}"; shift 2
mkdir -p "$RUN"
trap 'echo "[sbatch] USR1 received: graceful stop requested"; touch "$RUN/STOP"' USR1
# Count *allocated* GPUs: some Amarel nodes expose every GPU to nvidia-smi regardless of --gres.
NGPU="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "$NGPU" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l); fi
NGPU="${NGPU:-1}"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU - 1 ))
echo "[sbatch] restart count ${SLURM_RESTART_COUNT:-0}, $NGPU GPUs, $WORKERS loader workers per GPU"
torchrun --standalone --nproc_per_node="$NGPU" -m src.v3.train.run "$STAGE" --run-dir "$RUN" --workers "$WORKERS" "$@" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; done
wait "$PID"; code=$?
echo "[sbatch] trainer exited with $code"
exit $code
