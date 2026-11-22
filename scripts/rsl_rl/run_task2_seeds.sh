#!/usr/bin/env bash
# Run the task2 multi-seed experiments (seeds default to 40 41 42 43 44).
#
#   scripts/rsl_rl/run_task2_seeds.sh scratch|moe|componet|residual|fame [SEED ...]
#
# Runs one seed after another; a run (or chain) whose final model already exists is
# skipped, so the script can be re-launched after a crash. A chain that crashed
# midway restarts that seed from its first task (moe resumes instead).
# ROOT is the log root from scripts/common/log_root.py (override with CRL_LOG_ROOT); EXP is its
# folder name, used to tag KAE sessions and FAME chain checkpoints of this experiment.
#   scratch  : every task trained independently -> ROOT/<task>/scratch/<seed>
#   moe      : one KAE-MoE chain per seed (KAE session EXP_seed<seed>)
#              -> ROOT/<task>/moe/<seed>_EXP_seed<seed>
#   componet : one CompoNet chain per seed   -> ROOT/<task>/componet/<seed>
#   residual : one residual chain per seed   -> ROOT/<task>/residual/<seed>
#   fame     : one FAME chain per seed       -> ROOT/<task>/FAME/<seed>
#              (chain checkpoints in logs/skrl/FAME_CRL/EXP_seed<seed>)
# Network sizes come from agents/shared_net_cfg.py and the per-task budget from the
# rsl_rl runner configs, for every method.
# The moe chain needs KAEs/Base-Rough-Walking_KAE.pth (scripts/rsl_rl/fit_base_kae.py).
# Extra arguments for the train scripts go in TRAIN_ARGS, e.g. TRAIN_ARGS="--num_envs 1024".
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
METHOD="${1:?usage: $0 scratch|moe|componet|residual|fame [SEED ...]}"
shift
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(40 41 42 43 44)

ROOT="${CRL_LOG_ROOT:-$("$PYTHON" -c 'import sys; sys.path.insert(0, "scripts"); from common.log_root import DEFAULT_LOG_ROOT; print(DEFAULT_LOG_ROOT)')}"
export CRL_LOG_ROOT="$ROOT"   # every train script below reads it
EXP="$(basename "$ROOT")"
echo "[task2] log root: $ROOT"

# gym task id -> experiment_name (task folder under the log root), in the task2 order.
TASKS=(
  "Less-AnymalC-Jump-Direct-v1:anymal_c_jump_flat"
  "Less-Leg-Flat-Walking-Direct-v1:less_leg_walking_flat"
  "Less-AnymalC-Rough-Walking-Direct-v1:anymal_c_rough_leg_walking"
  "Less-AnymalC-Jump-Rough-Direct-v1:anymal_c_jump_rough"
  "Less-Leg-Rough-Walking-Direct-v1:less_leg_walking_rough"
)

case "$METHOD" in
  scratch)
    for seed in "${SEEDS[@]}"; do
      for entry in "${TASKS[@]}"; do
        task="${entry%%:*}"; name="${entry##*:}"
        if [ -f "$ROOT/$name/scratch/$seed/complete_model_with_metadata.pth" ]; then
          echo "[task2] skip scratch $name seed $seed (done)"; continue
        fi
        echo "[task2] scratch $name seed $seed  $(date '+%F %T')"
        # shellcheck disable=SC2086
        ALGO_METHOD=nominal "$PYTHON" scripts/rsl_rl/train_scratch.py \
          --task "$task" --seed "$seed" --headless $TRAIN_ARGS
      done
    done
    ;;
  moe)
    KAE_DIR=source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/KAEs
    [ -f "$KAE_DIR/Base-Rough-Walking_KAE.pth" ] || {
      echo "Missing $KAE_DIR/Base-Rough-Walking_KAE.pth; run scripts/rsl_rl/fit_base_kae.py --headless first." >&2
      exit 1
    }
    for seed in "${SEEDS[@]}"; do
      session="${EXP}_seed$seed"
      last="${TASKS[-1]##*:}"
      if [ -f "$ROOT/$last/moe/${seed}_${session}/complete_model_with_metadata.pth" ]; then
        echo "[task2] skip moe seed $seed (done)"; continue
      fi
      resume=()
      [ -d "$KAE_DIR/sessions/$session" ] && resume=(--resume)
      echo "[task2] moe seed $seed ${resume[*]:-}  $(date '+%F %T')"
      # shellcheck disable=SC2086
      "$PYTHON" scripts/rsl_rl/train_moe_CRL.py --seed "$seed" --session "$session" "${resume[@]}" \
        --headless $TRAIN_ARGS
    done
    ;;
  componet|residual)
    if [ "$METHOD" = componet ]; then algo=compo; driver=train_componet_CRL.py; else algo=res; driver=train_res_CRL.py; fi
    last="${TASKS[-1]##*:}"
    for seed in "${SEEDS[@]}"; do
      if [ -f "$ROOT/$last/$METHOD/$seed/complete_model_with_metadata.pth" ]; then
        echo "[task2] skip $METHOD seed $seed (done)"; continue
      fi
      echo "[task2] $METHOD seed $seed  $(date '+%F %T')"
      # shellcheck disable=SC2086
      ALGO_METHOD=$algo "$PYTHON" scripts/rsl_rl/$driver --seed "$seed" --headless $TRAIN_ARGS
    done
    ;;
  fame)
    BASE=source/less_leg_walking_1/less_leg_walking_1/tasks/direct/less_leg_walking_1/base_models/rough_walking.pth
    last="${TASKS[-1]##*:}"
    for seed in "${SEEDS[@]}"; do
      if [ -f "$ROOT/$last/FAME/$seed/checkpoint_final.pt" ]; then
        echo "[task2] skip fame seed $seed (done)"; continue
      fi
      echo "[task2] fame seed $seed  $(date '+%F %T')"
      # Per-seed checkpoint dir, so chains of different seeds never hand each other checkpoints.
      # shellcheck disable=SC2086
      "$PYTHON" scripts/skrl/train_FAME_CRL.py --seed "$seed" --init_policy "$BASE" \
        --ckpt_dir "logs/skrl/FAME_CRL/${EXP}_seed$seed" --headless $TRAIN_ARGS
    done
    ;;
  *)
    echo "unknown method: $METHOD (expected scratch, moe, componet, residual or fame)" >&2; exit 2 ;;
esac
echo "[task2] $METHOD finished: seeds ${SEEDS[*]}"
