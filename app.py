import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Quant Portfolio", layout="wide")
st.title("💼 Quant Portfolio Analytics Terminal")

# =========================================================
# THEME (SOFT DARK FINANCE)
# =========================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #0b1220);
    color: #e5e7eb;
}

/* SIDEBAR (AHORA BIEN INCLUIDO) */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* KPIs */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# COLORS
# =========================================================
colors = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#A855F7"]

# =========================================================
# SIDEBAR (PORTFOLIO BUILDER)
# =========================================================
st.sidebar.header("📊 Portfolio Builder")

tickers = st.sidebar.multiselect(
    "Assets",
    ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BTC-USD","ETH-USD"],
    default=["AAPL","MSFT","NVDA"]
)

start = st.sidebar.date_input("Start date", pd.to_datetime("2023-01-01"))
end = st.sidebar.date_input("End date", pd.to_datetime("today"))

show_rsi = st.sidebar.checkbox("Show RSI (optional)", True)

# Weights
weights = []
if tickers:
    st.sidebar.subheader("⚖️ Weights")

    for t in tickers:
        weights.append(st.sidebar.slider(t, 0.0, 1.0, 1/len(tickers)))

    weights = np.array(weights)
    weights = weights / weights.sum()

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return data["Close"]

if tickers:
    close = load_data(tickers, start, end)

    if isinstance(close, pd.Series):
        close = close.to_frame()

    returns = close.pct_change().dropna()

if not tickers or close.empty:
    st.stop()

# =========================================================
# PORTFOLIO METRICS
# =========================================================
port = returns @ weights
cum = (1 + port).cumprod()

ret = cum.iloc[-1] - 1
vol = port.std() * np.sqrt(252)
sharpe = port.mean() / port.std() * np.sqrt(252)
dd = (cum / cum.cummax() - 1).min()

c1, c2, c3, c4 = st.columns(4)

c1.metric("📈 Return", f"{ret:.2%}")
c2.metric("⚡ Volatility", f"{vol:.2%}")
c3.metric("🎯 Sharpe", f"{sharpe:.2f}")
c4.metric("📉 Max Drawdown", f"{dd:.2%}")

# =========================================================
# STYLE FUNCTION
# =========================================================
def style(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

# =========================================================
# PERFORMANCE
# =========================================================
st.subheader("📈 Performance")

fig = go.Figure()

for i, t in enumerate(tickers):
    s = close[t]
    perf = (s / s.iloc[0] - 1) * 100

    fig.add_trace(go.Scatter(
        x=s.index,
        y=perf,
        name=t,
        line=dict(color=colors[i % len(colors)], width=2)
    ))

st.plotly_chart(style(fig), use_container_width=True)

# =========================================================
# CORRELATION
# =========================================================
st.subheader("🌡️ Correlation")

fig = px.imshow(returns.corr(), text_auto=True, zmin=-1, zmax=1)
st.plotly_chart(style(fig), use_container_width=True)

# =========================================================
# ALLOCATION
# =========================================================
st.subheader("⚖️ Allocation")

fig = px.pie(names=tickers, values=weights, hole=0.5,
             color_discrete_sequence=colors)

st.plotly_chart(style(fig), use_container_width=True)

# =========================================================
# DRAWDOWN
# =========================================================
st.subheader("📉 Drawdown")

roll = cum.cummax()
dd_series = (cum - roll) / roll

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dd_series.index,
    y=dd_series * 100,
    fill="tozeroy",
    line=dict(color="#EF4444")
))

st.plotly_chart(style(fig), use_container_width=True)

# =========================================================
# MARKOWITZ OPTIMIZATION
# =========================================================
st.subheader("🧠 Optimal Portfolio (Sharpe Max)")

mean = returns.mean()
cov = returns.cov()
n = len(tickers)

def stats(w):
    r = np.dot(w, mean) * 252
    v = np.sqrt(np.dot(w.T, np.dot(cov * 252, w)))
    return r, v, r/v

def neg_sharpe(w):
    return -stats(w)[2]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
bounds = [(0,1)] * n
init = [1/n] * n

opt = minimize(neg_sharpe, init, bounds=bounds, constraints=cons)
w_opt = opt.x

opt_r, opt_v, opt_s = stats(w_opt)

c1, c2, c3 = st.columns(3)
c1.metric("Opt Return", f"{opt_r:.2%}")
c2.metric("Opt Vol", f"{opt_v:.2%}")
c3.metric("Opt Sharpe", f"{opt_s:.2f}")

fig = px.bar(x=tickers, y=w_opt, color=w_opt, color_continuous_scale="Viridis")
st.plotly_chart(style(fig), use_container_width=True)