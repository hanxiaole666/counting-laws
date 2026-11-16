"""VERI-FinRisk reproducibility suite.

All numbers consumed by the manuscript are produced here. No example data,
no hardcoded metrics, no silent fallbacks.
"""
import os

# This host ships an anaconda TensorFlow whose DLLs cannot load; transformers
# aborts while importing `transformers.trainer` because it eagerly probes the
# TF integration. Both flags must be set BEFORE transformers is imported, so
# they live at package import time.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "1.1.0"
