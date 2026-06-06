"""
NIFTY-50 AI Investment Intelligence Platform.

Modular pipeline for loading NSE historical data, engineering features,
and (in later phases) training models and building portfolio recommendations.
"""

__version__ = "0.2.0"

from src import anomaly, config, data_loader, features, portfolio, risk

__all__ = [
    "__version__",
    "anomaly",
    "config",
    "data_loader",
    "features",
    "portfolio",
    "risk",
]
