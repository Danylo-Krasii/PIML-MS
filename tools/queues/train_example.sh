#!/usr/bin/env bash
# Example queue for unattended training with the recipe of the released reach-13 checkpoints: for
# each seed in SEEDS, one nonlinear and one linear run on the corpus WHICH (cu, z5 or z8; defaults
# 0 and cu). Needs an NVIDIA GPU, nvidia-smi, the environment from `uv sync --extra cu118` and the
# corpus in data/periodic32/$WHICH (tools/README.md). On an RTX 4070 Laptop GPU with 8 GB the plain
# reach-13 runs took 120 to 211 min each, so the two default runs take about 4 to 7 h. Refuses to
# start while nvidia-smi lists a compute process, and ends with the number of failed runs. Under
# WSL2 nvidia-smi does not report compute processes, so there this check cannot see a busy GPU.
#
# Runs from the checkout it sits in, or from PIML_ROOT when that is set, and expects .venv/ there.
# Launch from the repository root:
#     mkdir -p logs && setsid nohup tools/queues/train_example.sh > logs/train_example.log 2>&1 < /dev/null &
# Another corpus or more seeds:
#     mkdir -p logs && WHICH=z8 SEEDS="0 1 2" setsid nohup tools/queues/train_example.sh > logs/train_z8.log 2>&1 < /dev/null &
cd "${PIML_ROOT:-$(dirname "$0")/../..}" || exit 1
PY=.venv/bin/python
WHICH="${WHICH:-cu}"
SEEDS="${SEEDS:-0}"
RECIPE="--width 64 --max-steps 36000 --augment --norm none --patience 6000 --sched-patience 20"
fail=0

run () {
    local label="$1"; shift
    echo "=== $(date -Is) START $label"
    "$PY" -m "$@"
    local rc=$?
    echo "=== $(date -Is) END $label rc=$rc"
    [ "$rc" = "0" ] || fail=$((fail + 1))
}

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) \
    || { echo "=== $(date -Is) nvidia-smi failed: not starting"; exit 1; }
if [ -n "$apps" ]; then echo "=== $(date -Is) GPU busy, compute pids: $apps; not starting"; exit 1; fi

for s in $SEEDS; do
    run "$WHICH-nonlinear-rf13-w64-s$s" piml.periodic sweep --which "$WHICH" --rf 13 $RECIPE --seeds "$s"
    run "$WHICH-linear-rf13-w64-s$s"    piml.periodic sweep --which "$WHICH" --linear --rf 13 $RECIPE --seeds "$s"
done

if [ "$fail" = "0" ]; then
    echo "=== $(date -Is) TRAIN EXAMPLE QUEUE COMPLETE"
else
    echo "=== $(date -Is) TRAIN EXAMPLE QUEUE FINISHED WITH $fail FAILED RUNS"
fi
