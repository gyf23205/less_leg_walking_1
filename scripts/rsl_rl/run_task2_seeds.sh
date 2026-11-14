#!/usr/bin/env bash
# Run the task2 multi-seed experiments for the scratch and KAE-MoE methods.
#
#   scripts/rsl_rl/run_task2_seeds.sh scratch [SEED ...]   # default seeds: 40 41 42 43 44
#   scripts/rsl_rl/run_task2_seeds.sh moe     [SEED ...]
#
# Runs one seed after another; a run that already has complete_model_with_metadata.pth
# is skipped, so the script can be re-launched after a crash.
#   scratch : every task trained independently -> logs/task2/<task>/scratch/<seed>
#   moe     : one KAE-MoE chain per seed (session task2_seed<seed>)
#             -> logs/task2/<task>/moe/<seed>_task2_seed<seed>
# The moe chain needs KAEs/Base-Rough-Walking_KAE.pth (scripts/rsl_rl/fit_base_kae.py).
# Extra arguments for the train scripts go in TRAIN_ARGS, e.g. TRAIN_ARGS="--num_envs 1024".
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
METHOD="${1:?usage: $0 scratch|moe [SEED ...]}"
shift
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(40 41 42 43 44)

# gym task id -> experiment_name (task folder under logs/task2), in the task2 order.
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
        if [ -f "logs/task2/$name/scratch/$seed/complete_model_with_metadata.pth" ]; then
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
      session="task2_seed$seed"
      last="${TASKS[-1]##*:}"
      if [ -f "logs/task2/$last/moe/${seed}_${session}/complete_model_with_metadata.pth" ]; then
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
  *)
    echo "unknown method: $METHOD (expected scratch or moe)" >&2; exit 2 ;;
esac
echo "[task2] $METHOD finished: seeds ${SEEDS[*]}"
