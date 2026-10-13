#!/usr/bin/env python3
"""Hierarchical design. Example:
python scripts/run_hier.py --group task --tasks translation,recall,continuation --langs ru,de --ks 4,8 --betas 0.3,1,3 --taus 1
"""
from prefixlab.cli_hier import main

if __name__ == "__main__":
    main()
