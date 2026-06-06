# src/config.py
"""
Configuration settings for the NIFTY-50 AI Platform.
Handles directory paths, data schemas, logging configurations,
and dynamic symbol discovery.
"""

import os
import logging
from typing import List, Dict

# Setup package-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("NIFTY_AI_Config")

# Directory Paths
# Resolving absolute paths based on this file's position (src/config.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
NIFTY50_DIR = os.path.join(DATA_DIR, "nifty50")
INDEX_DIR = os.path.join(DATA_DIR, "archive (1)", "Datasets", "INDEX")

METADATA_FILE = os.path.join(NIFTY50_DIR, "stock_metadata.csv")

# Expected columns and datatypes for equity data
EQUITY_SCHEMA = {
    'Date': 'datetime64[ns]',
    'Prev Close': 'float64',
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Last': 'float64',
    'Close': 'float64',
    'VWAP': 'float64',
    'Volume': 'float64',
    'Turnover': 'float64',
    'Trades': 'float64',
    'Deliverable Volume': 'float64',
    '%Deliverble': 'float64',
}

# Dynamic Stock Symbol and Sector Discovery
def discover_symbols_and_sectors() -> tuple[List[str], Dict[str, str]]:
    """
    Dynamically loads stock symbols and their corresponding industries from metadata.
    If metadata file is missing or invalid, discovers symbols from CSV filenames in nifty50 directory.

    Returns:
        tuple[List[str], Dict[str, str]]: A list of discovered symbols and a dictionary 
                                          mapping symbol -> industry classification.
    """
    symbols: List[str] = []
    sector_map: Dict[str, str] = {}

    # 1. Attempt loading from stock_metadata.csv
    if os.path.exists(METADATA_FILE):
        try:
            import csv
            with open(METADATA_FILE, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get("Symbol")
                    industry = row.get("Industry")
                    if symbol:
                        symbols.append(symbol)
                        if industry:
                            sector_map[symbol] = industry
            logger.info(f"Loaded {len(symbols)} symbols from metadata file: {METADATA_FILE}")
            return sorted(symbols), sector_map
        except Exception as e:
            logger.warning(f"Error reading metadata file {METADATA_FILE}: {e}. Falling back to file discovery.")

    # 2. Fallback: discover from filenames in directory
    if os.path.exists(NIFTY50_DIR):
        try:
            files = os.listdir(NIFTY50_DIR)
            for file in files:
                if file.endswith(".csv"):
                    # Exclude aggregated files and metadata file itself
                    if file in ("NIFTY50_all.csv", "stock_metadata.csv"):
                        continue
                    symbol = os.path.splitext(file)[0]
                    symbols.append(symbol)
                    sector_map[symbol] = "Unknown"  # Default sector since metadata is missing
            logger.info(f"Discovered {len(symbols)} symbols from filenames in: {NIFTY50_DIR}")
            return sorted(symbols), sector_map
        except Exception as e:
            logger.error(f"Error during directory scanning for symbols: {e}")
            
    return symbols, sector_map

# Load symbols and sectors dynamically at module import time
SYMBOLS, SECTOR_MAP = discover_symbols_and_sectors()

# Log error if no symbols could be found
if not SYMBOLS:
    logger.error("No NIFTY-50 symbols could be discovered. Check your raw data directories.")
