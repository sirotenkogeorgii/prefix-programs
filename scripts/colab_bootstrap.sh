#!/usr/bin/env bash
# Colab / Kaggle bootstrap. Run from a notebook cell:
#   !git clone https://<token>@github.com/sirotenkogeorgii/prefix-programs.git && cd prefix-programs && bash scripts/colab_bootstrap.sh
# Expects: HF_TOKEN in the environment (or `hf auth login` done), and optionally RESULTS_DIR pointing into Google Drive.
set -euo pipefail
cd "$(dirname "$0")/.."
pip uninstall -y -q torchao >/dev/null 2>&1 || true   # stale Colab torchao breaks peft import
pip install -q -e . >/dev/null
python - <<'PY'
import os, torch
from huggingface_hub import HfApi, login
tok = os.environ.get("HF_TOKEN")
if tok:
    login(token=tok, add_to_git_credential=False)
print("HF user:", HfApi().whoami()["name"])
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
# raw data: mLAMA (40 MB zip) and nothing else; FLORES+ streams from the Hub with the token
python -c "from prefixlab.data.mlama import ensure_mlama; print('mLAMA at', ensure_mlama('data_raw/mlama'))"
# results/checkpoints on Drive if RESULTS_DIR is set (symlinks keep the scripts' default paths working)
if [ -n "${RESULTS_DIR:-}" ]; then
  mkdir -p "$RESULTS_DIR/results" "$RESULTS_DIR/checkpoints"
  rm -rf results checkpoints
  ln -s "$RESULTS_DIR/results" results
  ln -s "$RESULTS_DIR/checkpoints" checkpoints
  echo "results -> $RESULTS_DIR/results"
fi
# a results dir with no tables yet (e.g. local disk after a Drive failure) is restored from the repo copy made by the sync cell
if [ -d results_sync ] && ! ls results/*.csv >/dev/null 2>&1; then
  mkdir -p results checkpoints
  cp results_sync/*.csv results/ 2>/dev/null || true
  [ -d results_sync/per_example ] && cp -r results_sync/per_example results/
  [ -d results_sync/checkpoints ] && cp -r results_sync/checkpoints/. checkpoints/
  echo "restored results and checkpoints from results_sync/"
fi
if ! ls results/*.csv >/dev/null 2>&1 && [ -n "${SEED_RESULTS_DIR:-}" ] && [ -d "$SEED_RESULTS_DIR/results" ]; then
  cp -r "$SEED_RESULTS_DIR/results/." results/ && cp -r "$SEED_RESULTS_DIR/checkpoints/." checkpoints/ 2>/dev/null || true
  echo "seeded results and checkpoints from $SEED_RESULTS_DIR"
fi
if ! ls results/*.csv >/dev/null 2>&1; then
  echo "WARNING: no results tables found; every run will start from scratch"
fi
python -m pytest -q -x 2>&1 | tail -1
