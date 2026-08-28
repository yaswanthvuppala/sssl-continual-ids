"""
download_anoshift.py — Root helper for AnoShift download/generation.
"""

import os
import sys

sys.path.append(os.path.abspath("ids-system"))
from data.download_anoshift import main

if __name__ == "__main__":
    main()
