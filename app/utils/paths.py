"""Project root and artifact path resolution for the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """Return the repository root (parent of ``app/``)."""
    return Path(__file__).resolve().parents[2]


def ensure_project_on_path() -> Path:
    """Insert project root on ``sys.path`` so ``src`` imports work."""
    root = project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


OUTPUTS_DIR_NAME = "outputs"

OOS_PREDICTIONS_FILE = "oos_predictions.parquet"
SUMMARY_METRICS_FILE = "summary_metrics.parquet"
FOLD_METRICS_FILE = "fold_metrics.parquet"
FEATURE_IMPORTANCE_FILE = "lgbm_feature_importance.parquet"


def outputs_dir() -> Path:
    return project_root() / OUTPUTS_DIR_NAME
