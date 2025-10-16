# src/gym_hpa/paths.py

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
