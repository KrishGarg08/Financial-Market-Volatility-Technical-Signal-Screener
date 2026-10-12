"""
app.py
Production-Ready Streamlit Financial Market Volatility & Technical Signal Screener
Clean, high-performance financial terminal interface powered by open-source libraries.
"""

from __future__ import annotations

import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics import fetch_ticker_data, compute_indicators, generate_signals, compute_risk_metrics

# Page configuration
st.set_page_config(
    page_title="AlphaPulse | Financial Volatility & Technical Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Financial Terminal CSS
st.markdown("""
<style>
    /* Global styling */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Terminal header */
    .terminal-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .terminal-sub {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }
    
    /* Metric Cards */
    .kpi-container {
        display: flex;
        gap: 12px;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 8px;
        padding: 14px 18px;
        flex: 1 1 200px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.75px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    
    .kpi-delta-pos {
        color: #10b981;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    .kpi-delta-neg {
        color: #ef4444;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    .kpi-tag {
        display: inline-block;
        font-size: 0.7rem;
        padding: 2px 7px;
        border-radius: 4px;
        margin-left: 6px;
        font-weight: 600;
    }
    
    .tag-bull {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .tag-bear {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .tag-neutral {
        background-color: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    /* Signal Cards */
    .signal-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    
    .signal-title {
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 2px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .signal-desc {
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.35;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.markdown("### ⚡ **AlphaPulse Terminal**")
st.sidebar.caption("Algorithmic Volatility & Signal Engine")

# Asset Selector
COMMON_TICKERS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL",
    "SPY", "QQQ", "IWM", "BTC-USD", "ETH-USD", "GLD", "USO"
]

selected_preset = st.sidebar.selectbox(
    "Asset Preset",
    options=["Select from list..."] + COMMON_TICKERS,
    index=1,  # Default to AAPL
    help="Select a high-liquidity stock, index ETF, or crypto pair."
)

custom_ticker = st.sidebar.text_input(
    "Or Enter Custom Ticker",
    value="" if selected_preset != "Select from list..." else "AAPL",
    help="Enter any valid Yahoo Finance symbol (e.g., AMD, COIN, ^VIX, SOL-USD)."
).strip().upper()

# Determine active symbol
active_symbol = custom_ticker if custom_ticker else (selected_preset if selected_preset != "Select from list..." else "AAPL")

# Timeframe & Interval Options
col_tf1, col_tf2 = st.sidebar.columns(2)
with col_tf1:
    timeframe = st.selectbox(
        "Timeframe",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=2,  # Default to 6mo
        help="Historical lookback window."
    )
with col_tf2:
    interval_options = ["1d", "1wk", "1mo"]
    if timeframe in ["1mo", "3mo"]:
        interval_options = ["1h", "1d", "1wk"]
    interval = st.selectbox(
        "Interval",
        options=interval_options,
        index=0 if "1d" in interval_options else 0,
        help="Bar aggregation period."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 **Indicator Overlays**")

toggle_sma = st.sidebar.checkbox("20 & 50 SMA (Simple MA)", value=True)
toggle_ema = st.sidebar.checkbox("20 & 50 EMA (Exponential MA)", value=False)
toggle_bb = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=True)
toggle_volume = st.sidebar.checkbox("Volume Bars", value=True)

st.sidebar.markdown("### 📉 **Subplot Oscillators**")
toggle_rsi = st.sidebar.checkbox("RSI (14-period)", value=True)
toggle_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=True)

st.sidebar.markdown("---")
refresh_data = st.sidebar.button("🔄 Refresh Market Data", use_container_width=True)
if refresh_data:
    st.cache_data.clear()

st.sidebar.caption("Data source: Free Yahoo Finance via yfinance. Zero external API fees.")


# ==========================================
# MAIN DASHBOARD CONTROLLER
# ==========================================
st.markdown(f"""
<div class="terminal-title">
    <span>📈 {active_symbol}</span>
    <span style="font-size: 1rem; font-weight: 400; color: #64748b;">| Lookback: {timeframe.upper()} | Interval: {interval.upper()}</span>
</div>
<div class="terminal-sub">Real-Time Algorithmic Technical Screener & Volatility Regime Analysis</div>
""", unsafe_allow_html=True)

try:
    with st.spinner(f"Fetching real-time market data for {active_symbol}..."):
        raw_df = fetch_ticker_data(active_symbol, period=timeframe, interval=interval)
        df = compute_indicators(raw_df)
        signal_results = generate_signals(df)
        risk_metrics = compute_risk_metrics(df)

except Exception as err:
    st.error(f"⚠️ Error retrieving or analyzing '{active_symbol}': {str(err)}")
    st.info("Tip: Please ensure ticker symbol matches Yahoo Finance convention (e.g. 'AAPL', 'BTC-USD', '^GSPC').")
    st.stop()


# ==========================================
# 1. TOP KPI METRICS ROW
# ==========================================
curr_row = df.iloc[-1]
prev_row = df.iloc[-2] if len(df) > 1 else curr_row

price = float(curr_row["Close"])
prev_price = float(prev_row["Close"])
daily_change = price - prev_price
daily_change_pct = (daily_change / prev_price * 100.0) if prev_price > 0 else 0.0

current_rsi = float(curr_row.get("RSI_14", 50.0))
rsi_status = "Overbought" if current_rsi >= 70 else ("Oversold" if current_rsi <= 30 else "Neutral")
rsi_tag_class = "tag-bear" if current_rsi >= 70 else ("tag-bull" if current_rsi <= 30 else "tag-neutral")

vol_30d = float(curr_row.get("Vol_30D_Ann", 0.0))
vol_21d = float(curr_row.get("Vol_21D_Ann", 0.0))
atr_14 = float(curr_row.get("ATR_14", 0.0))
atr_pct = (atr_14 / price * 100.0) if price > 0 else 0.0

vol_regime = "High" if vol_30d > 40 else ("Moderate" if vol_30d > 20 else "Low")
vol_tag_class = "tag-bear" if vol_regime == "High" else ("tag-neutral" if vol_regime == "Moderate" else "tag-bull")

high_52w = float(df["High"].max())
low_52w = float(df["Low"].min())

kpi_cols = st.columns(5)

with kpi_cols[0]:
    delta_class = "kpi-delta-pos" if daily_change >= 0 else "kpi-delta-neg"
    sign = "+" if daily_change >= 0 else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Current Price</div>
        <div class="kpi-value">${price:,.2f}</div>
        <div class="{delta_class}">{sign}${daily_change:,.2f} ({sign}{daily_change_pct:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">14D RSI Oscill.</div>
        <div class="kpi-value">{current_rsi:.1f} <span class="kpi-tag {rsi_tag_class}">{rsi_status}</span></div>
        <div class="terminal-sub" style="margin-bottom:0; font-size:0.75rem;">Momentum boundary: 30 / 70</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">30D Ann. Volatility</div>
        <div class="kpi-value">{vol_30d:.1f}% <span class="kpi-tag {vol_tag_class}">{vol_regime}</span></div>
        <div class="terminal-sub" style="margin-bottom:0; font-size:0.75rem;">21D Ann: {vol_21d:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">14D ATR Volatility</div>
        <div class="kpi-value">${atr_14:,.2f}</div>
        <div class="terminal-sub" style="margin-bottom:0; font-size:0.75rem;">Avg range: ±{atr_pct:.2f}% / bar</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[4]:
    range_pct = ((price - low_52w) / (high_52w - low_52w) * 100.0) if high_52w > low_52w else 50.0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Lookback Range</div>
        <div class="kpi-value">${low_52w:,.1f} – ${high_52w:,.1f}</div>
        <div class="terminal-sub" style="margin-bottom:0; font-size:0.75rem;">Position: {range_pct:.0f}% of range</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# 2. ALGORITHMIC SIGNAL SUMMARY BOX
# ==========================================
sentiment = signal_results.get("overall_sentiment", "Neutral")
score = signal_results.get("score", 0)
signals = signal_results.get("signals", [])

badge_color = "#10b981" if "Bullish" in sentiment else ("#ef4444" if "Bearish" in sentiment else "#f59e0b")
score_sign = "+" if score > 0 else ""

with st.expander("⚡ **Algorithmic Signal Matrix & Sentiment Breakdown**", expanded=True):
    col_sig_summary, col_sig_grid = st.columns([1, 2.5])
    
    with col_sig_summary:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 18px; text-align: center;">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 6px;">Consensus Algorithmic Rating</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {badge_color}; margin-bottom: 8px;">{sentiment.upper()}</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-bottom: 12px;">Net Signal Score: <span style="color: {badge_color}; font-size: 1.1rem;">{score_sign}{score}</span></div>
            <div style="display: flex; justify-content: space-around; border-top: 1px solid rgba(148, 163, 184, 0.15); padding-top: 10px; font-size: 0.8rem;">
                <span style="color: #10b981; font-weight: 600;">Bullish: {signal_results.get('bullish_count', 0)}</span>
                <span style="color: #ef4444; font-weight: 600;">Bearish: {signal_results.get('bearish_count', 0)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_sig_grid:
        # Render individual signal cards in 2 columns
        sig_cols = st.columns(2)
        for idx, sig in enumerate(signals):
            target_col = sig_cols[idx % 2]
            with target_col:
                tag_style = "tag-bull" if "Bullish" in sig["sentiment"] else ("tag-bear" if "Bearish" in sig["sentiment"] else "tag-neutral")
                target_col.markdown(f"""
                <div class="signal-card">
                    <div class="signal-title">
                        <span>{sig['badge']}</span>
                        <span class="kpi-tag {tag_style}">{sig['sentiment']}</span>
                    </div>
                    <div class="signal-desc">{sig['details']}</div>
                </div>
                """, unsafe_allow_html=True)


# ==========================================
# 2B. QUANTITATIVE RISK & PERFORMANCE MATRIX
# ==========================================
with st.expander("🛡️ **Quantitative Risk & Portfolio Performance Profile**", expanded=False):
    r_cols = st.columns(6)
    with r_cols[0]:
        st.metric(
            "Sharpe Ratio",
            f"{risk_metrics.get('sharpe_ratio', 0.0):.2f}",
            help="Annualized risk-adjusted excess return over 4% risk-free rate."
        )
    with r_cols[1]:
        st.metric(
            "Sortino Ratio",
            f"{risk_metrics.get('sortino_ratio', 0.0):.2f}",
            help="Annualized return scaled by downside semi-deviation."
        )
    with r_cols[2]:
        mdd_val = risk_metrics.get('max_drawdown', 0.0)
        st.metric(
            "Max Drawdown",
            f"{mdd_val:.1f}%",
            delta=f"{mdd_val:.1f}%",
            delta_color="inverse",
            help="Peak-to-trough decline over selected lookback window."
        )
    with r_cols[3]:
        st.metric(
            "95% Daily VaR",
            f"{risk_metrics.get('var_95_daily', 0.0):.2f}%",
            help="Maximum expected single-day loss at 95% historical confidence."
        )
    with r_cols[4]:
        cum_ret = risk_metrics.get('cumulative_return', 0.0)
        st.metric(
            "Period Return",
            f"{cum_ret:+.2f}%",
            delta=f"{cum_ret:+.2f}%",
            help="Cumulative holding period return."
        )
    with r_cols[5]:
        st.metric(
            "Daily Win Rate",
            f"{risk_metrics.get('win_rate', 0.0):.1f}%",
            help="Percentage of trading days with positive close-to-close returns."
        )


# ==========================================
# 3. INTERACTIVE PLOTLY FINANCIAL CHARTS
# ==========================================
st.markdown("### 📊 **Interactive Technical Chart**")

# Calculate row structure and heights dynamically
subplot_rows = 1  # Candlestick
row_specs = [{"secondary_y": False}]
row_heights = [0.60]

if toggle_volume:
    subplot_rows += 1
    row_specs.append({"secondary_y": False})
    row_heights.append(0.14)

if toggle_macd:
    subplot_rows += 1
    row_specs.append({"secondary_y": False})
    row_heights.append(0.18)

if toggle_rsi:
    subplot_rows += 1
    row_specs.append({"secondary_y": False})
    row_heights.append(0.16)

# Normalize row heights to sum to 1.0
total_h = sum(row_heights)
norm_heights = [h / total_h for h in row_heights]

chart_titles = ["Price & Overlays"]
if toggle_volume:
    chart_titles.append("Volume")
if toggle_macd:
    chart_titles.append("MACD (12, 26, 9)")
if toggle_rsi:
    chart_titles.append("RSI (14)")

fig = make_subplots(
    rows=subplot_rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.035,
    row_heights=norm_heights,
    subplot_titles=chart_titles,
)

current_chart_row = 1

# --- Row 1: Candlestick & Overlays ---
candlestick = go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name=f"{active_symbol} Price",
    increasing_line_color="#10b981",
    decreasing_line_color="#ef4444",
    increasing_fillcolor="#10b981",
    decreasing_fillcolor="#ef4444",
)
fig.add_trace(candlestick, row=current_chart_row, col=1)

# Bollinger Bands Overlays
if toggle_bb and "BB_Upper" in df.columns:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Upper"],
        line=dict(color="rgba(147, 197, 253, 0.4)", width=1, dash="dot"),
        name="BB Upper (2σ)",
        hoverinfo="skip"
    ), row=current_chart_row, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Lower"],
        line=dict(color="rgba(147, 197, 253, 0.4)", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.05)",
        name="BB Lower (2σ)",
        hoverinfo="skip"
    ), row=current_chart_row, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Middle"],
        line=dict(color="rgba(147, 197, 253, 0.8)", width=1.2),
        name="BB Mid (20 SMA)",
    ), row=current_chart_row, col=1)

# SMA Overlays
if toggle_sma:
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"],
            line=dict(color="#f59e0b", width=1.5),
            name="20 SMA"
        ), row=current_chart_row, col=1)
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"],
            line=dict(color="#3b82f6", width=1.5),
            name="50 SMA"
        ), row=current_chart_row, col=1)

# EMA Overlays
if toggle_ema:
    if "EMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_20"],
            line=dict(color="#ec4899", width=1.5, dash="dash"),
            name="20 EMA"
        ), row=current_chart_row, col=1)
    if "EMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_50"],
            line=dict(color="#8b5cf6", width=1.5, dash="dash"),
            name="50 EMA"
        ), row=current_chart_row, col=1)

# --- Row 2: Volume Bars ---
if toggle_volume:
    current_chart_row += 1
    vol_colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color=vol_colors,
        opacity=0.6,
    ), row=current_chart_row, col=1)

# --- Row 3: MACD Subplot ---
if toggle_macd:
    current_chart_row += 1
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        line=dict(color="#38bdf8", width=1.8),
        name="MACD"
    ), row=current_chart_row, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"],
        line=dict(color="#fb923c", width=1.5, dash="dash"),
        name="Signal (9D)"
    ), row=current_chart_row, col=1)

    hist_colors = ["#10b981" if h >= 0 else "#ef4444" for h in df["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"],
        marker_color=hist_colors,
        name="MACD Hist",
        opacity=0.75,
    ), row=current_chart_row, col=1)

# --- Row 4: RSI Subplot ---
if toggle_rsi:
    current_chart_row += 1
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI_14"],
        line=dict(color="#a855f7", width=2),
        name="RSI (14)"
    ), row=current_chart_row, col=1)

    # Threshold guidelines
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", line_width=1, row=current_chart_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", line_width=1, row=current_chart_row, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(148, 163, 184, 0.3)", line_width=1, row=current_chart_row, col=1)

# Styling Plotly Layout
chart_total_height = 450 + (160 * (subplot_rows - 1))
fig.update_layout(
    height=chart_total_height,
    template="plotly_dark",
    paper_bgcolor="rgba(15, 23, 42, 0.5)",
    plot_bgcolor="rgba(15, 23, 42, 0.5)",
    margin=dict(l=15, r=15, t=30, b=15),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11),
    ),
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(148, 163, 184, 0.08)",
    zeroline=False,
    showline=True,
    linecolor="rgba(148, 163, 184, 0.2)",
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148, 163, 184, 0.08)",
    zeroline=False,
    showline=True,
    linecolor="rgba(148, 163, 184, 0.2)",
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 4. HISTORICAL DATA PREVIEW & CSV EXPORT
# ==========================================
st.markdown("### 📋 **Historical Data & Metric Inspector**")

# Prepare export dataframe with rounded readable metrics
export_df = df.copy()
export_df.index = export_df.index.strftime("%Y-%m-%d")
display_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "BB_Bandwidth", "ATR_14", "Vol_21D_Ann", "Vol_30D_Ann"
]
available_display_cols = [c for c in display_cols if c in export_df.columns]
clean_preview = export_df[available_display_cols].sort_index(ascending=False).round(2)

col_tbl, col_dl = st.columns([4, 1])
with col_dl:
    csv_bytes = clean_preview.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Export Full Dataset (CSV)",
        data=csv_bytes,
        file_name=f"{active_symbol}_{timeframe}_{interval}_analytics_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download calculated technical metrics and indicator time-series to CSV."
    )

st.dataframe(
    clean_preview.head(50),
    use_container_width=True,
    height=300
)

# Terminal Footer
st.markdown("---")
st.caption("AlphaPulse Quantitative Market Screener | Pure Open-Source Stack: Streamlit • yfinance • pandas • plotly • pandas-ta")
