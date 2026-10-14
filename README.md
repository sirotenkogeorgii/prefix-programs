# Programs for a Frozen Machine

Soft prefixes as programs for a frozen multilingual language model: how many bits, and how many prefix slots,
does it take to tell the model which language to produce in and which task to perform, and are the two
separable? See `Prefix_Project_Proposal_v1.md` for the design and hypotheses.

## Layout

- `prefixlab/` — scoring (teacher-forced bits), soft / variational / hierarchical / composed prefixes, prompt tuning,
  LoRA floors, prequential coding, warm-start distances, regime-2 input compression, window patching, analysis.
- `scripts/run_sweep.py` — Design A sweep (`--design flat` for the variational rate), baselines, floors.
- `scripts/run_hier.py` — hierarchical design (`--group task` or `--group lang`).
- `scripts/run_compose.py` — tied task + language parts with held-out cells (composition cost, zero-shot composition).
- `scripts/run_distance.py` — warm-start prequential distance between all cells at one k (prefixes from `results/sweep.csv`).
- `scripts/run_regime2.py` — regime 2: per-example input compression into m vectors, with or without the shared prefix.
- `scripts/run_mechanism.py` — per-block window patching of the prefix positions (Design A, hierarchical centre and deviation).
- `scripts/colab_bootstrap.sh` — install, log in, fetch mLAMA, link results to Drive, run tests.
- `results/*.csv` — one row per finished run; every script skips rows already present. `results/per_example/` holds
  per-example test bits for bootstrap intervals.
- `tests/` — pytest on a tiny random Qwen2 config (no model download).

## Quick start

```bash
pip install -e .
export HF_TOKEN=...   # FLORES+ is gated: accept its terms on the Hub once
python scripts/run_sweep.py --model Qwen/Qwen2.5-1.5B --tasks translation,recall,continuation --langs ru,de \
    --ks 1,2,4,8,16,32 --seeds 0,1 --baselines --floors --results results/sweep.csv
python -m pytest
```

Data: FLORES+ (CC BY-SA 4.0, gated) and mLAMA (CC BY-NC-SA 4.0). Model: Qwen2.5 base checkpoints.
