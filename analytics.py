"""
analytics.py
Market Volatility & Technical Signal Analytics Engine
Provides market data fetching, indicator computations, and rule-based algorithmic trading signals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Configure logger
logger = logging.getLogger(__name__)

# Try importing pandas_ta safely
try:
    import pandas_ta as ta  # type: ignore
    HAS_PANDAS_TA = True
except Exception as e:
    logger.warning("pandas_ta could not be imported (%s). Pure pandas/numpy fallbacks will be used.", e)
    HAS_PANDAS_TA = False


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given ticker symbol using yfinance.
    Implements robust error handling, caching, column cleanup, and timezone normalization.

    Parameters:
        symbol: Ticker symbol (e.g. 'AAPL', 'BTC-USD', '^GSPC')
        period: Valid yfinance period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        interval: Data interval ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo')

    Returns:
        pd.DataFrame: Cleaned OHLCV DataFrame with DateTime index.
    """
    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("Ticker symbol cannot be empty.")

    try:
        ticker = yf.Ticker(clean_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            # Fallback to yf.download in case history fails on certain formats
            df = yf.download(clean_symbol, period=period, interval=interval, progress=False, auto_adjust=True)

        if df is None or df.empty:
            raise ValueError(f"No price data found for ticker '{clean_symbol}'. Please verify the symbol.")

        # Handle multi-level columns if returned by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize required OHLCV column names
        col_rename = {col: col.capitalize() for col in df.columns}
        df = df.rename(columns=col_rename)

        required_cols = ["Open", "High", "Low", "Close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required price columns: {missing_cols}")

        if "Volume" not in df.columns:
            df["Volume"] = 0

        # Remove rows where Close is NaN
        df = df.dropna(subset=["Close"]).copy()

        # Normalize datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # If timezone-aware, convert to tz-naive or keep clean ISO format
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.sort_index(ascending=True, inplace=True)

        if len(df) < 5:
            raise ValueError(f"Insufficient historical data points ({len(df)}) for '{clean_symbol}'.")

        return df

    except Exception as exc:
        logger.error("Error fetching data for ticker %s: %s", clean_symbol, exc)
        raise RuntimeError(f"Failed to fetch market data for '{clean_symbol}': {str(exc)}") from exc


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators and volatility metrics:
    - 20-day & 50-day Simple Moving Average (SMA_20, SMA_50)
    - 20-day & 50-day Exponential Moving Average (EMA_20, EMA_50)
    - Relative Strength Index (RSI_14)
    - MACD Line, Signal Line (9-day), and MACD Histogram
    - Bollinger Bands: Upper, Middle (SMA 20), Lower, Bandwidth, and %B
    - Average True Range (ATR_14)
    - Annualized Rolling Volatility (21-day and 30-day log returns)

    Parameters:
        df: pd.DataFrame containing Open, High, Low, Close, Volume.

    Returns:
        pd.DataFrame: Augmented DataFrame with computed technical indicator columns.
    """
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    # 1. Moving Averages (SMA & EMA)
    data["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    data["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    data["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    # 2. Relative Strength Index (RSI, 14-period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's Exponential Smoothing for RSI
    alpha_rsi = 1.0 / 14.0
    avg_gain = gain.ewm(alpha=alpha_rsi, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha_rsi, min_periods=14, adjust=False).mean()
    
    # Mathematical edge case: zero loss -> RSI = 100; zero gain -> RSI = 0
    safe_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / safe_loss
    rsi_computed = 100.0 - (100.0 / (1.0 + rs))
    
    # When avg_loss is 0, if avg_gain > 0, RSI is 100.0
    data["RSI_14"] = np.where(
        (avg_loss == 0) & (avg_gain > 0),
        100.0,
        np.where(
            (avg_gain == 0) & (avg_loss > 0),
            0.0,
            rsi_computed
        )
    )
    data["RSI_14"] = pd.Series(data["RSI_14"], index=data.index).fillna(50.0)


    # 3. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # 4. Bollinger Bands (20-period, 2 standard deviations)
    bb_window = 20
    bb_std_mult = 2.0
    data["BB_Middle"] = close.rolling(window=bb_window, min_periods=1).mean()
    rolling_std = close.rolling(window=bb_window, min_periods=1).std(ddof=0)
    data["BB_Upper"] = data["BB_Middle"] + (bb_std_mult * rolling_std)
    data["BB_Lower"] = data["BB_Middle"] - (bb_std_mult * rolling_std)
    
    # Bollinger Bandwidth and %B
    denom = data["BB_Middle"].replace(0, np.nan)
    data["BB_Bandwidth"] = (data["BB_Upper"] - data["BB_Lower"]) / denom
    band_range = (data["BB_Upper"] - data["BB_Lower"]).replace(0, np.nan)
    data["BB_Pct"] = (close - data["BB_Lower"]) / band_range

    # 5. Average True Range (ATR, 14-period)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["True_Range"] = true_range
    alpha_atr = 1.0 / 14.0
    data["ATR_14"] = true_range.ewm(alpha=alpha_atr, min_periods=14, adjust=False).mean()

    # 6. Annualized Rolling Volatility (21-day & 30-day log returns)
    log_return = np.log(close / close.shift(1))
    data["Log_Return"] = log_return
    
    # 21-day rolling volatility annualized (standard 252 trading days)
    data["Vol_21D_Ann"] = log_return.rolling(window=21, min_periods=5).std(ddof=1) * np.sqrt(252) * 100.0
    # 30-day rolling volatility annualized
    data["Vol_30D_Ann"] = log_return.rolling(window=30, min_periods=5).std(ddof=1) * np.sqrt(252) * 100.0

    # Backfill earliest NaN values with first available metric for graceful UI rendering
    cols_to_bfill = ["SMA_20", "SMA_50", "EMA_20", "EMA_50", "BB_Upper", "BB_Middle", "BB_Lower", "ATR_14", "Vol_21D_Ann", "Vol_30D_Ann"]
    for c in cols_to_bfill:
        if c in data.columns:
            data[c] = data[c].bfill()

    return data


def generate_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates rule-based algorithmic trading signals based on the latest market indicators:
    - RSI Momentum & Overbought/Oversold boundaries
    - MACD Crossovers (Bullish / Bearish within recent candles) & Histogram momentum
    - Bollinger Bands: Squeeze detection, Breakouts (Upper/Lower), Mean-reversion
    - Moving Average Trends: SMA 20/50 Golden Cross & Death Cross, EMA trend alignment
    - Volatility Regime: ATR expansion/contraction vs rolling baseline

    Parameters:
        df: pd.DataFrame enriched by compute_indicators.

    Returns:
        Dict[str, Any] containing individual signal details, consensus score, and overall sentiment.
    """
    if len(df) < 5:
        return {
            "overall_sentiment": "Neutral",
            "score": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "signals": [],
        }

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else prev1

    signals: List[Dict[str, str]] = []
    bullish_pts = 0
    bearish_pts = 0

    # --- 1. RSI Signal ---
    rsi = float(curr.get("RSI_14", 50.0))
    if rsi >= 70.0:
        bearish_pts += 1
        signals.append({
            "category": "RSI Indicator",
            "sentiment": "Bearish",
            "badge": "🔴 RSI Overbought",
            "details": f"14-Day RSI is {rsi:.1f} (≥ 70.0). Asset is heavily overbought; elevated risk of short-term pullback.",
            "status": "Warning"
        })
    elif rsi <= 30.0:
        bullish_pts += 1
        signals.append({
            "category": "RSI Indicator",
            "sentiment": "Bullish",
            "badge": "🟢 RSI Oversold",
            "details": f"14-Day RSI is {rsi:.1f} (≤ 30.0). Asset is heavily oversold; high probability of mean-reverting relief bounce.",
            "status": "Opportunity"
        })
    elif 60.0 <= rsi < 70.0:
        signals.append({
            "category": "RSI Indicator",
            "sentiment": "Neutral / Alert",
            "badge": "⚠️ RSI Approaching Overbought",
            "details": f"14-Day RSI is {rsi:.1f}. Strong bullish momentum, but approaching caution territory.",
            "status": "Watch"
        })
    elif 30.0 < rsi <= 40.0:
        signals.append({
            "category": "RSI Indicator",
            "sentiment": "Neutral / Alert",
            "badge": "⚠️ RSI Approaching Oversold",
            "details": f"14-Day RSI is {rsi:.1f}. Downward pressure slowing, entering potential reversal zone.",
            "status": "Watch"
        })
    else:
        signals.append({
            "category": "RSI Indicator",
            "sentiment": "Neutral",
            "badge": "⚪ RSI Neutral",
            "details": f"14-Day RSI is balanced at {rsi:.1f} (normal range 40–60).",
            "status": "Neutral"
        })

    # --- 2. MACD Crossover & Momentum ---
    macd_curr = float(curr.get("MACD", 0.0))
    sig_curr = float(curr.get("MACD_Signal", 0.0))
    macd_prev = float(prev1.get("MACD", 0.0))
    sig_prev = float(prev1.get("MACD_Signal", 0.0))
    macd_hist = float(curr.get("MACD_Hist", 0.0))

    # Crossover detection within last 2 bars
    is_bullish_cross = (macd_prev <= sig_prev and macd_curr > sig_curr) or (macd_curr > sig_curr and float(prev2.get("MACD", 0.0)) <= float(prev2.get("MACD_Signal", 0.0)))
    is_bearish_cross = (macd_prev >= sig_prev and macd_curr < sig_curr) or (macd_curr < sig_curr and float(prev2.get("MACD", 0.0)) >= float(prev2.get("MACD_Signal", 0.0)))

    if is_bullish_cross:
        bullish_pts += 2
        signals.append({
            "category": "MACD Crossover",
            "sentiment": "Bullish",
            "badge": "🟢 Bullish MACD Crossover",
            "details": f"MACD line ({macd_curr:.3f}) recently crossed above the 9-day Signal line ({sig_curr:.3f}). Momentum turning positive.",
            "status": "Strong Buy"
        })
    elif is_bearish_cross:
        bearish_pts += 2
        signals.append({
            "category": "MACD Crossover",
            "sentiment": "Bearish",
            "badge": "🔴 Bearish MACD Crossover",
            "details": f"MACD line ({macd_curr:.3f}) recently crossed below the 9-day Signal line ({sig_curr:.3f}). Momentum turning negative.",
            "status": "Strong Sell"
        })
    elif macd_curr > sig_curr and macd_hist > 0:
        bullish_pts += 1
        signals.append({
            "category": "MACD Trend",
            "sentiment": "Bullish",
            "badge": "🟢 Bullish MACD Momentum",
            "details": f"MACD is above Signal line with expanding positive histogram ({macd_hist:+.3f}).",
            "status": "Active Bull"
        })
    elif macd_curr < sig_curr and macd_hist < 0:
        bearish_pts += 1
        signals.append({
            "category": "MACD Trend",
            "sentiment": "Bearish",
            "badge": "🔴 Bearish MACD Momentum",
            "details": f"MACD is below Signal line with negative histogram ({macd_hist:+.3f}).",
            "status": "Active Bear"
        })

    # --- 3. Bollinger Bands: Squeeze & Breakouts ---
    price = float(curr["Close"])
    bb_upper = float(curr.get("BB_Upper", price * 1.05))
    bb_lower = float(curr.get("BB_Lower", price * 0.95))
    bb_bw = float(curr.get("BB_Bandwidth", 0.1))

    # Calculate 60-day bandwidth percentile to detect squeeze
    recent_bw = df["BB_Bandwidth"].tail(60).dropna()
    is_squeeze = False
    if len(recent_bw) >= 10:
        bw_rank = (recent_bw < bb_bw).mean()
        if bw_rank <= 0.20:
            is_squeeze = True

    if price >= bb_upper:
        signals.append({
            "category": "Bollinger Bands",
            "sentiment": "Bullish / Extended",
            "badge": "⚡ Upper Bollinger Breakout",
            "details": f"Price (${price:,.2f}) pierced upper band (${bb_upper:,.2f}). Strong volatility expansion or overextension.",
            "status": "Breakout"
        })
        bullish_pts += 1
    elif price <= bb_lower:
        signals.append({
            "category": "Bollinger Bands",
            "sentiment": "Bearish / Extended",
            "badge": "⚡ Lower Bollinger Breakdown",
            "details": f"Price (${price:,.2f}) pierced lower band (${bb_lower:,.2f}). Downside breakdown or extreme discount.",
            "status": "Breakdown"
        })
        bearish_pts += 1
    elif is_squeeze:
        signals.append({
            "category": "Bollinger Bands",
            "sentiment": "Neutral / Alert",
            "badge": "🟡 Bollinger Band Squeeze",
            "details": f"Bandwidth ({bb_bw:.3%}) is near multi-week lows. Volatility compression signals an imminent explosive breakout.",
            "status": "Squeeze Watch"
        })
    else:
        signals.append({
            "category": "Bollinger Bands",
            "sentiment": "Neutral",
            "badge": "⚪ Within Bollinger Range",
            "details": f"Price is trading safely inside the volatility envelope (${bb_lower:,.2f} – ${bb_upper:,.2f}).",
            "status": "Normal"
        })

    # --- 4. Moving Average Trend Alignment ---
    sma_20 = float(curr.get("SMA_20", price))
    sma_50 = float(curr.get("SMA_50", price))
    ema_20 = float(curr.get("EMA_20", price))
    ema_50 = float(curr.get("EMA_50", price))

    sma20_prev = float(prev1.get("SMA_20", sma_20))
    sma50_prev = float(prev1.get("SMA_50", sma_50))

    # Golden Cross / Death Cross detection
    if sma20_prev <= sma50_prev and sma_20 > sma_50:
        bullish_pts += 2
        signals.append({
            "category": "Moving Averages",
            "sentiment": "Bullish",
            "badge": "🏆 Golden Cross (SMA 20/50)",
            "details": f"20-day SMA (${sma_20:,.2f}) recently crossed above 50-day SMA (${sma_50:,.2f}). Major medium-term bullish confirmation.",
            "status": "Golden Cross"
        })
    elif sma20_prev >= sma50_prev and sma_20 < sma_50:
        bearish_pts += 2
        signals.append({
            "category": "Moving Averages",
            "sentiment": "Bearish",
            "badge": "💀 Death Cross (SMA 20/50)",
            "details": f"20-day SMA (${sma_20:,.2f}) crossed below 50-day SMA (${sma_50:,.2f}). Major medium-term bearish trend signal.",
            "status": "Death Cross"
        })
    elif price > ema_20 > sma_50:
        bullish_pts += 1
        signals.append({
            "category": "Moving Averages",
            "sentiment": "Bullish",
            "badge": "🟢 Bullish Trend Structure",
            "details": f"Price (${price:,.2f}) > 20 EMA (${ema_20:,.2f}) > 50 SMA (${sma_50:,.2f}). Uptrend structure well supported.",
            "status": "Bullish Trend"
        })
    elif price < ema_20 < sma_50:
        bearish_pts += 1
        signals.append({
            "category": "Moving Averages",
            "sentiment": "Bearish",
            "badge": "🔴 Bearish Trend Structure",
            "details": f"Price (${price:,.2f}) < 20 EMA (${ema_20:,.2f}) < 50 SMA (${sma_50:,.2f}). Asset under systematic distribution.",
            "status": "Bearish Trend"
        })
    else:
        signals.append({
            "category": "Moving Averages",
            "sentiment": "Neutral",
            "badge": "⚪ Choppy MA Alignment",
            "details": f"Moving averages are consolidating without a distinct directional trend (SMA20: ${sma_20:,.2f}, SMA50: ${sma_50:,.2f}).",
            "status": "Consolidation"
        })

    # --- 5. Volatility & ATR Regime ---
    atr = float(curr.get("ATR_14", 0.0))
    atr_hist = df["ATR_14"].tail(20).dropna()
    avg_atr = float(atr_hist.mean()) if len(atr_hist) > 0 else atr
    ann_vol_21 = float(curr.get("Vol_21D_Ann", 0.0))

    if atr > (1.35 * avg_atr) and avg_atr > 0:
        signals.append({
            "category": "Volatility Regime",
            "sentiment": "Alert",
            "badge": "⚠️ ATR Volatility Expansion",
            "details": f"Current 14D ATR (${atr:,.2f}) is 35%+ higher than its 20-day baseline (${avg_atr:,.2f}). Heightened risk regime.",
            "status": "High Volatility"
        })
    elif atr < (0.75 * avg_atr) and avg_atr > 0:
        signals.append({
            "category": "Volatility Regime",
            "sentiment": "Alert",
            "badge": "🧊 ATR Volatility Contraction",
            "details": f"Current 14D ATR (${atr:,.2f}) is 25%+ below baseline. Quiet market environment preceding volatility expansion.",
            "status": "Low Volatility"
        })
    else:
        signals.append({
            "category": "Volatility Regime",
            "sentiment": "Neutral",
            "badge": "📊 Normal Volatility",
            "details": f"14D ATR is ${atr:,.2f} with 21-day annualized volatility at {ann_vol_21:.1f}%.",
            "status": "Stable"
        })

    # Consensus scoring
    net_score = bullish_pts - bearish_pts
    if net_score >= 3:
        overall = "Strongly Bullish"
    elif net_score in (1, 2):
        overall = "Bullish"
    elif net_score == 0:
        overall = "Neutral"
    elif net_score in (-1, -2):
        overall = "Bearish"
    else:
        overall = "Strongly Bearish"

    return {
        "overall_sentiment": overall,
        "score": net_score,
        "bullish_count": bullish_pts,
        "bearish_count": bearish_pts,
        "signals": signals,
    }


def compute_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.04) -> Dict[str, float]:
    """
    Computes institutional risk and performance analytics:
    - Cumulative Return (%)
    - Annualized Return (CAGR %)
    - Annualized Volatility (%)
    - Sharpe Ratio (Annualized)
    - Sortino Ratio (Downside deviation annualized)
    - Maximum Drawdown (MDD %)
    - 95% Daily Value at Risk (Parametric & Historical VaR %)
    - Win Rate (% Positive Trading Days)
    - Profit Factor (Gross Gain / Gross Loss)

    Parameters:
        df: pd.DataFrame containing 'Close' prices and DateTime index.
        risk_free_rate: Annual benchmark risk-free rate (e.g. 0.04 for 4%).

    Returns:
        Dict[str, float]: Quantitative risk and return metrics.
    """
    if len(df) < 5 or "Close" not in df.columns:
        return {
            "cumulative_return": 0.0,
            "cagr": 0.0,
            "annualized_vol": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "var_95_daily": 0.0,
            "win_rate": 0.0,
            "profit_factor": 1.0,
        }

    close = df["Close"]
    daily_returns = close.pct_change().dropna()
    if len(daily_returns) == 0:
        return {}

    # Cumulative return
    cum_return = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0

    # Number of trading days & CAGR
    n_days = len(daily_returns)
    years = max(n_days / 252.0, 1.0 / 252.0)
    cagr = ((close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0) * 100.0

    # Annualized volatility
    daily_vol = float(daily_returns.std(ddof=1))
    ann_vol = daily_vol * np.sqrt(252) * 100.0

    # Sharpe ratio
    rf_daily = risk_free_rate / 252.0
    excess_returns = daily_returns - rf_daily
    sharpe = float((excess_returns.mean() / daily_vol) * np.sqrt(252)) if daily_vol > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = float(downside_returns.std(ddof=1)) if len(downside_returns) > 1 else daily_vol
    ann_downside_vol = downside_std * np.sqrt(252)
    sortino = float(((cagr / 100.0) - risk_free_rate) / ann_downside_vol) if ann_downside_vol > 0 else 0.0

    # Maximum Drawdown (MDD)
    cum_peaks = close.cummax()
    drawdowns = (close - cum_peaks) / cum_peaks * 100.0
    max_dd = float(drawdowns.min())

    # 95% Daily Value at Risk (Historical percentile)
    var_95_hist = float(-np.percentile(daily_returns, 5) * 100.0)

    # Win rate
    positive_days = (daily_returns > 0).sum()
    win_rate = float(positive_days / n_days * 100.0)

    # Profit Factor
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else 1.0

    return {
        "cumulative_return": round(cum_return, 2),
        "cagr": round(cagr, 2),
        "annualized_vol": round(ann_vol, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_dd, 2),
        "var_95_daily": round(var_95_hist, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
    }

