# src/data_loader.py
"""
Data Ingestion and Alignment Module for NIFTY-50 Stock Market Data.
Provides functions to load stock, index, and metadata CSVs, and aligns them on
a common trading day timeline.
"""

import os
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from src.config import (
    DATA_DIR,
    NIFTY50_DIR,
    INDEX_DIR,
    METADATA_FILE,
    EQUITY_SCHEMA,
    resolve_canonical_symbol,
    resolve_symbol_filename,
)

logger = logging.getLogger("NIFTY_AI_DataLoader")

def load_metadata() -> pd.DataFrame:
    """
    Loads stock metadata mapping symbols to industry classification.

    Returns:
        pd.DataFrame: Metadata DataFrame.
    """
    logger.info("Loading stock metadata...")
    if not os.path.exists(METADATA_FILE):
        error_msg = f"Metadata file not found at: {METADATA_FILE}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        df = pd.read_csv(METADATA_FILE)
        logger.info(f"Successfully loaded metadata with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load metadata file: {e}")
        raise e


def load_stock_data(symbol: str) -> pd.DataFrame:
    """
    Loads individual historical stock daily records.

    Args:
        symbol (str): Equity ticker symbol (e.g., 'ADANIPORTS', 'M&M', or 'MM').

    Returns:
        pd.DataFrame: Cleaned daily historical dataframe.
    """
    canonical = resolve_canonical_symbol(symbol)
    filename_stem = resolve_symbol_filename(symbol)
    file_path = os.path.join(NIFTY50_DIR, f"{filename_stem}.csv")
    if not os.path.exists(file_path):
        logger.warning(
            f"File for stock '{symbol}' (canonical: '{canonical}', "
            f"expected file: '{filename_stem}.csv') not found at {file_path}"
        )
        return pd.DataFrame()

    try:
        # Load and parse Date column
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Enforce column schema types where present
        for col, dtype in EQUITY_SCHEMA.items():
            if col in df.columns:
                if col == 'Date':
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        
        # Sort values chronologically
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(
            f"Loaded stock '{canonical}' from {filename_stem}.csv with shape {df.shape}"
        )
        return df
    except Exception as e:
        logger.error(f"Error loading data for symbol {symbol}: {e}")
        return pd.DataFrame()


def load_index_data(index_name: str) -> pd.DataFrame:
    """
    Loads historical index data (e.g., 'NIFTY 50' or 'INDIA VIX').

    Gracefully returns an empty DataFrame when the archive dataset or a
    specific index file is missing, with a clear log message.

    Args:
        index_name (str): Filename without extension in INDEX folder.

    Returns:
        pd.DataFrame: Cleaned index dataframe, or empty if data is unavailable.
    """
    archive_dir = os.path.join(DATA_DIR, "archive (1)")
    if not os.path.isdir(archive_dir):
        logger.warning(
            f"Archive data directory not found at '{archive_dir}'. "
            f"Cannot load index '{index_name}'. "
            "Extract or copy the archive dataset to data/archive (1)/Datasets/INDEX/."
        )
        return pd.DataFrame()

    if not os.path.isdir(INDEX_DIR):
        logger.warning(
            f"Index directory not found at '{INDEX_DIR}'. "
            f"Cannot load index '{index_name}'. "
            "Expected CSV files such as 'NIFTY 50.csv' and 'INDIA VIX.csv'."
        )
        return pd.DataFrame()

    file_path = os.path.join(INDEX_DIR, f"{index_name}.csv")
    if not os.path.exists(file_path):
        available = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(INDEX_DIR)
            if f.lower().endswith(".csv")
        )
        hint = f" Available indices: {', '.join(available[:8])}" if available else ""
        if len(available) > 8:
            hint += f" (and {len(available) - 8} more)"
        logger.warning(
            f"Index file '{index_name}.csv' not found at '{file_path}'.{hint}"
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(f"Loaded index '{index_name}' with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(
            f"Failed to parse index '{index_name}' from '{file_path}': {e}"
        )
        return pd.DataFrame()


def create_aligned_panel(
    symbols: List[str], 
    start_date: str = "2000-01-03", 
    end_date: str = "2021-04-30"
) -> pd.DataFrame:
    """
    Joins all stock close prices onto a common trading day calendar calendar.
    Calendar dates are extracted from the NIFTY 50 benchmark index.
    Late listing symbols are kept with NaN values prior to their listings.
    
    Args:
        symbols (List[str]): List of stock symbols to align.
        start_date (str): Beginning of date index range.
        end_date (str): End of date index range.

    Returns:
        pd.DataFrame: Multi-column dataframe of aligned stock close prices,
                      indexed by Date.
    """
    logger.info(f"Creating aligned panel of Close prices from {start_date} to {end_date}...")
    
    # 1. Load NIFTY 50 index to act as master calendar
    nifty50_df = load_index_data("NIFTY 50")
    if nifty50_df.empty:
        # Fallback to standard business date range if benchmark CSV fails
        logger.warning("NIFTY 50 index empty. Falling back to pandas date range.")
        date_idx = pd.date_range(start=start_date, end=end_date, freq='B')
    else:
        # Mask calendar within date range
        nifty50_df = nifty50_df[(nifty50_df['Date'] >= start_date) & (nifty50_df['Date'] <= end_date)]
        date_idx = nifty50_df['Date'].unique()
    
    # Create empty DataFrame indexed by the master date calendar
    aligned_df = pd.DataFrame(index=pd.DatetimeIndex(date_idx))
    aligned_df.index.name = 'Date'
    
    for symbol in symbols:
        stock_df = load_stock_data(symbol)
        if stock_df.empty:
            aligned_df[symbol] = np.nan
            continue
        
        # Keep Date and Close columns, set Date as index
        stock_df = stock_df[['Date', 'Close']].drop_duplicates(subset=['Date'])
        stock_df = stock_df.set_index('Date')
        
        # Reindex to master calendar
        # Note: Do NOT fill pre-listing periods. They should remain NaN.
        # But if there are intermittent missing values (e.g. trading halt) we can ffill.
        reindexed_stock = stock_df.reindex(aligned_df.index)
        aligned_df[symbol] = reindexed_stock['Close']
        
    logger.info(f"Aligned panel created successfully. Shape: {aligned_df.shape}")
    return aligned_df
