#!/usr/bin/env python3
"""Design A sweep. Example:
python scripts/run_sweep.py --model Qwen/Qwen2.5-1.5B --tasks translation,recall,continuation --langs ru,de --ks 1,2,4,8,16,32 --seeds 0,1 --baselines
"""
from prefixlab.cli_sweep import main

if __name__ == "__main__":
    main()
