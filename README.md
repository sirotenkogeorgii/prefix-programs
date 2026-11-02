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
  per-example test bits for bootstrap intervals, `results/history/` the per-eval-step (dev bits, KL) trajectory of each run.
- `results_sync/` — the committed copy of every result table, per-example file, history and prefix checkpoint from the
  Colab runs (the Colab notebook restores from it and the autosync cell pushes to it).
- `prefixlab/analysis/` — floors and normalized distortion, k* and savings tables, rate–distortion frontiers,
  hierarchical totals, distance matrix and silhouettes, mechanism recovery profiles, composition hold-outs, regime 2,
  bootstrap intervals, the common-uuid recall check, and the figures.
- `notebooks/01_colab_runner.ipynb` — runs the whole grid on Colab (A100, ~32 h), resumable.
- `notebooks/01_figures.ipynb` — regenerates `figures/` and `tables/` from `results_sync/` and `prereg.json`, no GPU.
- `notebooks/02_live_demo.ipynb` — trains one prefix live and places it on the stored curve (GPU).
- `report/report.md` (+ PDF) and `slides/slides.md` (+ PDF, PPTX) — the seminar report and talk, built with pandoc.
- `paper/paper.tex` (+ PDF, `refs.bib`, NeurIPS style file) — the paper-format write-up, built with `latexmk -pdf`.
- `prereg.json` — operating points, grids and training settings fixed on dev before any test number.
- `tests/` — pytest on a tiny random Qwen2 config (no model download).

## Quick start

```bash
pip install -e .
export HF_TOKEN=...   # FLORES+ is gated: accept its terms on the Hub once
python scripts/run_sweep.py --model Qwen/Qwen2.5-1.5B --tasks translation,recall,continuation --langs ru,de \
    --ks 1,2,4,8,16,32 --seeds 0,1 --baselines --floors --results results/sweep.csv
python -m pytest
# figures and tables from the committed results, no GPU:
(cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_figures.ipynb)
python scripts/draw_sampling_figure.py   # schematic of the two units and the draw schedule (figures/sampling.png)
# post-review jobs (GPU): language groups retrained on measurable members, and zero-rate baselines (no training):
python scripts/run_hier.py --group lang --tasks translation,recall --langs ru,de --ks 2,4 --betas 1,3 --protocol corrected --lr 0.01 \
    --results results/hier2_lang_meas.csv --ckpt-dir checkpoints/hier2_lang_meas
python scripts/run_baseline.py --ks 2,4 --kinds prior,vocab --n-draws 8 --results results/baseline.csv
# report and slides:
(cd report && pandoc report.md -o report.pdf --pdf-engine=xelatex --resource-path=.:..)
(cd slides && pandoc slides.md -t beamer -o slides.pdf --pdf-engine=xelatex --resource-path=.:.. -V sansfont="Helvetica Neue")
```

Variational runs: `--design flat` is the preregistered protocol; `--design flat2 --lr 0.01` the corrected one
(checkpoint by dev objective, dev scored with posterior samples), which is the headline in the report.
Hierarchical runs: `scripts/run_hier.py --protocol prereg` (default) writes `hier_task.csv` / `hier_lang.csv`;
`--protocol corrected --lr 0.01` writes `hier2_task.csv` / `hier2_lang.csv`, the like-for-like comparison with flat2.

Data: FLORES+ (CC BY-SA 4.0, gated) and mLAMA (CC BY-NC-SA 4.0). Model: Qwen2.5 base checkpoints.
