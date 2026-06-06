# NIFTY-50 AI Investment Intelligence Platform

An AI-powered decision-support platform for NIFTY-50 constituent stocks and market indices (2000–2021). The project ingests historical NSE equity and index data, engineers technical and macro features, and will eventually deliver medium-term alpha signals, risk-aware portfolio construction, and explainable recommendations via a Streamlit dashboard.

## Dataset Description

The workspace uses three primary data sources:

1. **Constituent equities** (`data/nifty50/`)
   - One CSV per NIFTY-50 stock (e.g. `INFY.csv`, `MM.csv` for Mahindra & Mahindra).
   - Columns: `Date`, `Symbol`, `Series`, OHLC, `VWAP`, `Volume`, `Turnover`, `Trades`, deliverable volume fields.
   - `NIFTY50_all.csv` — vertically stacked records for all constituents.
   - `stock_metadata.csv` — symbol-to-industry mapping (canonical symbols such as `M&M`).

2. **Market indices & volatility** (`data/archive (1)/Datasets/INDEX/`)
   - `NIFTY 50.csv` — benchmark index (master trading calendar).
   - `INDIA VIX.csv` — volatility index (from July 2010).
   - Sector indices (e.g. `NIFTY IT.csv`, `NIFTY BANK.csv`) for relative-strength features.

3. **Full NSE scrip archive** (`data/archive (1)/Datasets/SCRIP/`) — optional extended universe (gitignored by default).

> **Note:** The `data/archive (1)/` folder is listed in `.gitignore`. Copy or extract the archive dataset locally before running index-dependent notebooks or features.

### Symbol aliases

Some metadata symbols differ from on-disk filenames. For example, metadata lists `M&M` but the file is `MM.csv`. The `src.config` module resolves these via `resolve_canonical_symbol()` and `resolve_symbol_filename()`.

## Current Project Status

| Phase | Scope | Status |
|-------|--------|--------|
| **Phase 1** | Data loaders, EDA notebook | Complete |
| **Phase 2** | Feature engineering (`features.py`), feature analysis notebook | Complete |
| **Phase 3** | Portfolio optimization, risk engine, backtesting | Not started |
| **Phase 4** | SHAP explainability, Streamlit dashboard, report | Not started |

Implemented modules:

- `src/config.py` — paths, schemas, symbol discovery, aliases
- `src/data_loader.py` — stock/index loading, aligned price panels
- `src/features.py` — technical, momentum, risk, and index overlays
- `src/anomaly.py` — placeholder for future anomaly detection

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design and roadmap.

## Folder Structure

```
NIFTY-AI-Platform/
├── data/
│   ├── nifty50/                  # NIFTY-50 constituent CSVs + metadata
│   └── archive (1)/
│       └── Datasets/
│           ├── INDEX/            # Benchmark & sector indices (local only)
│           └── SCRIP/            # Full NSE scrip history (local only)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   └── anomaly.py
├── notebooks/
│   ├── 1.0_exploratory_analysis.ipynb
│   └── 2.0_feature_analysis.ipynb
├── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository** and enter the project root:

   ```bash
   git clone <repository-url>
   cd NIFTY-AI-Platform
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Place datasets locally**:
   - Ensure `data/nifty50/` contains constituent CSVs and `stock_metadata.csv`.
   - Extract the archive bundle to `data/archive (1)/Datasets/INDEX/` (and optionally `SCRIP/`) for index and VIX features.

## Running Notebooks

From the project root, launch Jupyter:

```bash
jupyter notebook
```

Open notebooks under `notebooks/`. Each notebook prepends the parent directory to `sys.path` so `src` imports work when the kernel cwd is `notebooks/`:

- `1.0_exploratory_analysis.ipynb` — Phase 1 EDA (metadata, missing values, sectors, correlations, regimes).
- `2.0_feature_analysis.ipynb` — Phase 2 feature matrix, visualizations, collinearity, leakage checks.

Alternatively, register the project root as the Jupyter kernel working directory or install the package in editable mode if you add a `pyproject.toml` later.

## Future Roadmap

```
Phase 1 (EDA) ──> Phase 2 (Features) ──> Phase 3 (Portfolio & Risk) ──> Phase 4 (Streamlit & XAI)
```

- **Phase 3** — `portfolio.py`, `risk.py`: MVO with covariance shrinkage, Black-Litterman, VaR/CVaR, drawdown metrics, backtests with transaction costs.
- **Phase 4** — `explainers.py`, `app/dashboard.py`: SHAP explanations, interactive Streamlit prototype, technical report.

Planned additions not yet in the repo: `models.py`, `portfolio.py`, `risk.py`, `explainers.py`, `app/dashboard.py`.

## License

See repository license terms if provided by the project owner.
