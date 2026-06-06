# src/features.py
"""
Feature Engineering Module for the NIFTY-50 AI Platform.
Computes technical, momentum, risk, and market indicators from equity and index data.
Enforces zero lookahead bias.
"""

import os
import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from src.config import SYMBOLS, SECTOR_MAP
from src.data_loader import load_stock_data, load_index_data

logger = logging.getLogger("NIFTY_AI_Features")

# Mapping stock sector metadata to index files inside INDEX directory
SECTOR_INDEX_MAP = {
    'FINANCIAL SERVICES': 'NIFTY FIN SERVICE',
    'ENERGY': 'NIFTY ENERGY',
    'AUTOMOBILE': 'NIFTY AUTO',
    'CONSUMER GOODS': 'NIFTY FMCG',
    'METALS': 'NIFTY METAL',
    'PHARMA': 'NIFTY PHARMA',
    'IT': 'NIFTY IT',
    'MEDIA & ENTERTAINMENT': 'NIFTY MEDIA',
    'SERVICES': 'NIFTY INFRA',
    'CONSTRUCTION': 'NIFTY INFRA',
    'TELECOM': 'NIFTY 50',                  # Fallback to market benchmark
    'CEMENT & CEMENT PRODUCTS': 'NIFTY 50', # Fallback to market benchmark
    'FERTILISERS & PESTICIDES': 'NIFTY 50', # Fallback to market benchmark
}

# --- Core Indicator Calculations ---

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates standard Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's EMA smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates MACD Line, Signal Line, and MACD Histogram."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculates Bollinger Upper, Lower, Bandwidth, and %B values."""
    sma = series.rolling(window=period).mean()
    rstd = series.rolling(window=period).std()
    upper = sma + num_std * rstd
    lower = sma - num_std * rstd
    bandwidth = (upper - lower) / (sma + 1e-9)
    percent_b = (series - lower) / (upper - lower + 1e-9)
    return upper, lower, bandwidth, percent_b


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def calculate_rolling_beta(stock_ret: pd.Series, market_ret: pd.Series, window: int = 60) -> pd.Series:
    """Calculates rolling systematic Beta coefficient."""
    cov = stock_ret.rolling(window=window).cov(market_ret)
    var = market_ret.rolling(window=window).var()
    return cov / (var + 1e-9)


# --- Master Feature Matrix Generator ---

def generate_stock_features(symbol: str, market_df: pd.DataFrame, vix_df: pd.DataFrame, sector_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Computes all technical, momentum, risk, and index features for a single stock.
    Returns the DataFrame containing Date, Symbol, close, and feature columns.
    """
    df = load_stock_data(symbol)
    if df.empty:
        return pd.DataFrame()
    
    # 1. Clean price columns and returns
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Technical Indicators (Stock Level)
    for w in [10, 20, 50, 200]:
        df[f"SMA_{w}"] = df['Close'].rolling(window=w).mean()
        df[f"EMA_{w}"] = df['Close'].ewm(span=w, adjust=False).mean()
        
    df["RSI_14"] = calculate_rsi(df['Close'], period=14)
    
    macd_l, signal_l, macd_h = calculate_macd(df['Close'])
    df["MACD_line"] = macd_l
    df["MACD_signal"] = signal_l
    df["MACD_hist"] = macd_h
    
    bb_up, bb_low, bb_w, bb_pct = calculate_bollinger_bands(df['Close'])
    df["BB_upper"] = bb_up
    df["BB_lower"] = bb_low
    df["BB_bandwidth"] = bb_w
    df["BB_pct_b"] = bb_pct
    
    # 3. Momentum Features
    for h in [5, 10, 21]:
        df[f"ROC_{h}"] = df['Close'] / df['Close'].shift(h) - 1.0
        
    df["Volume_Momentum"] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-9)
    
    # 4. Risk Features
    df["ATR_14"] = calculate_atr(df['High'], df['Low'], df['Close'], period=14)
    
    for w in [10, 20, 60]:
        df[f"Volatility_{w}"] = df['Returns'].rolling(window=w).std() * np.sqrt(252)
        
    # Align NIFTY-50 returns on dates to compute Beta and relative strength
    market_align = pd.merge(df[['Date', 'Returns', 'ROC_21']], market_df[['Date', 'Market_Ret', 'Market_ROC_21']], on='Date', how='left')
    df["Beta_60"] = calculate_rolling_beta(market_align['Returns'], market_align['Market_Ret'], window=60)
    df["Relative_Strength_21"] = market_align['ROC_21'] - market_align['Market_ROC_21']
    
    # 5. Sector Index returns
    industry = SECTOR_MAP.get(symbol, 'TELECOM') # Default to Telecom if sector missing
    sect_index = SECTOR_INDEX_MAP.get(industry, 'NIFTY 50')
    df["Sector_Ret"] = df['Date'].map(sector_returns.get(sect_index, sector_returns['NIFTY 50']))
    df["Sector_Momentum_5"] = df['Date'].map(sector_returns.get(sect_index + "_Mom_5", sector_returns['NIFTY 50_Mom_5']))
    df["Sector_Momentum_21"] = df['Date'].map(sector_returns.get(sect_index + "_Mom_21", sector_returns['NIFTY 50_Mom_21']))
    
    # 6. INDIA VIX level and rate of change (VIX change rate)
    vix_align = pd.merge(df[['Date']], vix_df[['Date', 'Close', 'VIX_ROC_5']], on='Date', how='left')
    df["VIX_level"] = vix_align['Close']
    df["VIX_ROC_5"] = vix_align['VIX_ROC_5']
    
    # Prepare standard columns
    feature_cols = [c for c in df.columns if c not in ['Date', 'Symbol', 'Series']]
    # Rename features to standard format: {symbol}_{feature_name}
    renamed_df = pd.DataFrame()
    renamed_df['Date'] = df['Date']
    for col in feature_cols:
        renamed_df[f"{symbol}_{col}"] = df[col]
        
    return renamed_df


def build_feature_matrix(symbols: List[str], start_date: str = "2000-01-03", end_date: str = "2021-04-30") -> pd.DataFrame:
    """
    Builds and merges all constituent stock features onto a common Date index.
    
    Args:
        symbols (List[str]): List of stock symbols to compute features for.
        start_date (str): Timeline start.
        end_date (str): Timeline end.

    Returns:
        pd.DataFrame: Complete master feature matrix.
    """
    logger.info("Initializing Master Feature Matrix Construction...")
    
    # 1. Load Market Index (NIFTY 50) and calculate returns
    nifty_df = load_index_data("NIFTY 50")
    if nifty_df.empty:
        raise ValueError("NIFTY 50 benchmark data is empty.")
    nifty_df['Market_Ret'] = np.log(nifty_df['Close'] / nifty_df['Close'].shift(1))
    nifty_df['Market_ROC_21'] = nifty_df['Close'] / nifty_df['Close'].shift(21) - 1.0
    nifty_df['Market_Mom_5'] = nifty_df['Close'] / nifty_df['Close'].shift(5) - 1.0
    nifty_df['Market_Mom_21'] = nifty_df['Close'] / nifty_df['Close'].shift(21) - 1.0
    
    # 2. Load INDIA VIX and calculate 5-day rate of change
    vix_df = load_index_data("INDIA VIX")
    if not vix_df.empty:
        vix_df['VIX_ROC_5'] = vix_df['Close'] / vix_df['Close'].shift(5) - 1.0
    else:
        # Create dummy columns if missing
        vix_df = pd.DataFrame(columns=['Date', 'Close', 'VIX_ROC_5'])
    
    # 3. Load all unique sector indexes and calculate returns
    sector_returns: Dict[str, pd.Series] = {}
    for sect_idx in set(SECTOR_INDEX_MAP.values()):
        idx_df = load_index_data(sect_idx)
        if not idx_df.empty:
            idx_df['Sector_Ret'] = np.log(idx_df['Close'] / idx_df['Close'].shift(1))
            idx_df['Sector_Mom_5'] = idx_df['Close'] / idx_df['Close'].shift(5) - 1.0
            idx_df['Sector_Mom_21'] = idx_df['Close'] / idx_df['Close'].shift(21) - 1.0
            sector_returns[sect_idx] = pd.Series(idx_df['Sector_Ret'].values, index=idx_df['Date'])
            sector_returns[sect_idx + "_Mom_5"] = pd.Series(idx_df['Sector_Mom_5'].values, index=idx_df['Date'])
            sector_returns[sect_idx + "_Mom_21"] = pd.Series(idx_df['Sector_Mom_21'].values, index=idx_df['Date'])
            
    # Include NIFTY 50 returns as a fallback
    sector_returns['NIFTY 50'] = pd.Series(nifty_df['Market_Ret'].values, index=nifty_df['Date'])
    sector_returns['NIFTY 50_Mom_5'] = pd.Series(nifty_df['Market_Mom_5'].values, index=nifty_df['Date'])
    sector_returns['NIFTY 50_Mom_21'] = pd.Series(nifty_df['Market_Mom_21'].values, index=nifty_df['Date'])

    # 4. Generate and align stock features on master calendar
    nifty_df = nifty_df[(nifty_df['Date'] >= start_date) & (nifty_df['Date'] <= end_date)]
    master_dates = nifty_df['Date'].unique()
    
    master_df = pd.DataFrame(index=pd.DatetimeIndex(master_dates))
    master_df.index.name = 'Date'
    
    logger.info(f"Looping across {len(symbols)} symbols to calculate features...")
    for symbol in symbols:
        stock_feat = generate_stock_features(symbol, nifty_df, vix_df, sector_returns)
        if stock_feat.empty:
            continue
        
        # Merge on master dates index
        stock_feat = stock_feat.set_index('Date').reindex(master_df.index)
        # Drop Symbol column as it is wide-format
        stock_feat = stock_feat.drop(columns=['Symbol'], errors='ignore')
        
        # Join into master dataframe
        master_df = master_df.join(stock_feat, how='left')
        
    logger.info(f"Master feature matrix constructed successfully. Shape: {master_df.shape}")
    return master_df
