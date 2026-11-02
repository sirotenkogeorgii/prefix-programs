#!/usr/bin/env python3
"""Zero-rate baselines (no training): prefixes drawn from the embedding prior and random token rows. Example:
python scripts/run_baseline.py --model Qwen/Qwen2.5-1.5B --tasks translation,recall,continuation --langs ru,de,en \
    --ks 2,4 --kinds prior,vocab --n-draws 8 --results results/baseline.csv
"""
from prefixlab.cli_baseline import main

if __name__ == "__main__":
    main()
