#!/usr/bin/env bash
# Colab / Kaggle bootstrap. Run from a notebook cell:
#   !git clone https://<token>@github.com/sirotenkogeorgii/prefix-programs.git && cd prefix-programs && bash scripts/colab_bootstrap.sh
# Expects: HF_TOKEN in the environment (or `hf auth login` done), and optionally RESULTS_DIR pointing into Google Drive.
set -euo pipefail
cd "$(dirname "$0")/.."
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
python -m pytest -q -x 2>&1 | tail -1
