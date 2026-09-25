from __future__ import annotations

import math
from pathlib import Path
from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from portfolio_engine import build_benchmark_curve, portfolio_backtest
from robustness_engine import combination_count, heatmap_slice, parameter_sensitivity_summary, run_parameter_scan

from stock_engine import (
    MODEL_VERSION,
    add_indicators,
    apply_transaction_cost,
    backtest_stats,
    fetch_stock_data,
    fetch_stock_info,
    forward_ai_calibration,
    forward_performance_summary,
    normalize_stock_id,
    recent_support_resistance,
    recent_signal_log,
    signal_performance_stats,
    signal_expectancy_stats,
    strategy_lab_backtest,
    strength_label,
    technical_screen_row,
    train_prediction_model,
    v_signal_quality,
    watchlist_signal_row,
)


st.set_page_config(
    page_title="免費台股 AI 多空分析 v9",
    page_icon="📈",
    layout="wide",
)

# 免費版觀察池：刻意限制規模，避免一次大量 API 呼叫。
WATCHLISTS = {
    "大型權值": ["2330", "2317", "2454", "2308", "2382", "2881", "2882", "2891", "2886", "2412", "2303", "3711", "1216", "1301", "2002"],
    "AI / 電子": ["2330", "2454", "2382", "3231", "2376", "2357", "2356", "2377", "6669", "3443", "3661", "3034", "3017", "3653", "2345"],
    "金融": ["2881", "2882", "2886", "2891", "2892", "2884", "2885", "2880", "2883", "5871", "5880", "2887"],
    "ETF": ["0050", "0056", "00878", "00919", "00929", "006208", "00713", "00881", "00922", "00923"],
}

KNOWN_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電", "2382": "廣達",
    "2881": "富邦金", "2882": "國泰金", "2891": "中信金", "2886": "兆豐金", "2412": "中華電",
    "2303": "聯電", "3711": "日月光投控", "1216": "統一", "1301": "台塑", "2002": "中鋼",
    "3231": "緯創", "2376": "技嘉", "2357": "華碩", "2356": "英業達", "2377": "微星",
    "6669": "緯穎", "3443": "創意", "3661": "世芯-KY", "3034": "聯詠", "3017": "奇鋐",
    "3653": "健策", "2345": "智邦", "2892": "第一金", "2884": "玉山金", "2885": "元大金",
    "2880": "華南金", "2883": "凱基金", "5871": "中租-KY", "5880": "合庫金", "2887": "台新新光金",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00919": "群益台灣精選高息",
    "00929": "復華台灣科技優息", "006208": "富邦台50", "00713": "元大台灣高息低波", "00881": "國泰台灣科技龍頭",
    "00922": "國泰台灣領袖50", "00923": "群益台ESG低碳50",
}




def get_streamlit_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


@st.cache_data(ttl=60, show_spinner=False)
def load_forward_log(path_text: str = "data/forward_signals.csv") -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        table = pd.read_csv(path, dtype={"stock_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "data_date" in table.columns:
        table["data_date"] = pd.to_datetime(table["data_date"], errors="coerce")
    if "stock_id" in table.columns:
        table["stock_id"] = table["stock_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return table

@st.cache_data(ttl=60, show_spinner=False)
def load_notification_log(path_text: str = "data/notification_events.csv") -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        table = pd.read_csv(path, dtype={"stock_id": str, "event_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "data_date" in table.columns:
        table["data_date"] = pd.to_datetime(table["data_date"], errors="coerce")
    if "created_at" in table.columns:
        table["created_at"] = pd.to_datetime(table["created_at"], errors="coerce")
    return table


def load_small_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(stock_id: str, years: int, source: str, token: str):
    return fetch_stock_data(stock_id, years=years, source=source, finmind_token=token)


@st.cache_data(ttl=3600, show_spinner=False)
def prepare_data(df: pd.DataFrame, threshold: float, retrain_every: int):
    enriched = add_indicators(df)
    model_result = train_prediction_model(
        enriched,
        threshold=threshold,
        retrain_every=retrain_every,
    )
    return enriched, model_result


@st.cache_data(ttl=86400, show_spinner=False)
def load_stock_info(token: str):
    return fetch_stock_info(token)


def pct_text(x: float, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x) * 100:.{digits}f}%"


def no_weekend(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def price_chart(df: pd.DataFrame, days: int = 180) -> go.Figure:
    p = df.tail(days)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=p["date"], open=p["open"], high=p["high"], low=p["low"], close=p["close"],
            name="K線", increasing_line_color="#d62728", decreasing_line_color="#2ca02c",
        )
    )
    fig.add_trace(go.Scatter(x=p["date"], y=p["ma20"], name="MA20", mode="lines"))
    fig.add_trace(go.Scatter(x=p["date"], y=p["ma60"], name="MA60", mode="lines"))
    buy = p[p["signal"] == "V"]
    sell = p[p["signal"] == "A"]
    if not buy.empty:
        fig.add_trace(go.Scatter(
            x=buy["date"], y=buy["low"] * 0.98, mode="markers+text", text=["V"] * len(buy),
            textposition="bottom center", marker=dict(symbol="triangle-up", size=12), name="V 轉強",
        ))
    if not sell.empty:
        fig.add_trace(go.Scatter(
            x=sell["date"], y=sell["high"] * 1.02, mode="markers+text", text=["A"] * len(sell),
            textposition="top center", marker=dict(symbol="triangle-down", size=12), name="A 轉弱",
        ))
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False,
                      legend_orientation="h", hovermode="x unified")
    return no_weekend(fig)


def strength_candle_chart(df: pd.DataFrame, days: int = 180) -> go.Figure:
    p = df.tail(days).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.04)
    bands = [
        (70, 101, "強多", "#8b0000"),
        (50, 70, "偏多", "#ef8a62"),
        (30, 50, "偏空", "#91cf60"),
        (-1, 30, "強空", "#006837"),
    ]
    for lower, upper, label, color in bands:
        mask = (p["strength_score"] >= lower) & (p["strength_score"] < upper)
        q = p.loc[mask]
        if q.empty:
            continue
        fig.add_trace(
            go.Candlestick(
                x=q["date"], open=q["open"], high=q["high"], low=q["low"], close=q["close"],
                name=label, increasing_line_color=color, increasing_fillcolor=color,
                decreasing_line_color=color, decreasing_fillcolor=color,
            ), row=1, col=1,
        )

    # 趨勢翻轉線：EMA20 依多/空狀態分色。
    bull_line = p["ema20"].where(p["trend_state"] == 1)
    bear_line = p["ema20"].where(p["trend_state"] == -1)
    fig.add_trace(go.Scatter(x=p["date"], y=bull_line, mode="lines", name="翻轉線・多", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=p["date"], y=bear_line, mode="lines", name="翻轉線・空", line=dict(width=2)), row=1, col=1)

    buy = p[p["signal"] == "V"]
    sell = p[p["signal"] == "A"]
    if not buy.empty:
        fig.add_trace(go.Scatter(
            x=buy["date"], y=buy["low"] * 0.98, mode="markers+text", text=["V"] * len(buy),
            textposition="bottom center", marker=dict(symbol="triangle-up", size=13), name="V 轉強",
        ), row=1, col=1)
    if not sell.empty:
        fig.add_trace(go.Scatter(
            x=sell["date"], y=sell["high"] * 1.02, mode="markers+text", text=["A"] * len(sell),
            textposition="top center", marker=dict(symbol="triangle-down", size=13), name="A 轉弱",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(x=p["date"], y=p["strength_score"], mode="lines", name="力道分數"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1, title_text="力道")
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False,
                      legend_orientation="h", hovermode="x unified")
    return no_weekend(fig)


def indicator_chart(df: pd.DataFrame, days: int = 180) -> go.Figure:
    p = df.tail(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=p["date"], y=p["rsi14"], name="RSI(14)", mode="lines"), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", row=1, col=1)
    fig.add_trace(go.Scatter(x=p["date"], y=p["macd_hist"], name="MACD Histogram", mode="lines"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", legend_orientation="h")
    return no_weekend(fig)


def backtest_chart(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["strategy_curve"], name="Walk-forward 策略", mode="lines"))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["benchmark_curve"], name="買進持有", mode="lines"))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=35, b=10), yaxis_title="資產倍數",
                      legend_orientation="h", hovermode="x unified")
    return no_weekend(fig)


def strategy_equity_chart(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["equity"], name="策略資金曲線", mode="lines+markers"))
    fig.add_hline(y=1.0, line_dash="dot")
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis_title="資產倍數",
        hovermode="x unified",
    )
    return no_weekend(fig)


def portfolio_equity_chart(equity: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["equity"], name="V 訊號投資組合", mode="lines"))
    if benchmark is not None and not benchmark.empty:
        fig.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["equity"], name="0050 買進持有基準", mode="lines"))
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis_title="模擬資產",
        legend_orientation="h",
        hovermode="x unified",
    )
    return no_weekend(fig)


def portfolio_position_chart(equity: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
    if not equity.empty:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["positions"], name="持股檔數", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["cash_ratio"] * 100, name="現金比重%", mode="lines"), row=2, col=1)
    fig.update_yaxes(title_text="檔數", row=1, col=1)
    fig.update_yaxes(title_text="現金%", range=[0, 105], row=2, col=1)
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h", hovermode="x unified")
    return no_weekend(fig)


def parse_tickers(text: str) -> list[str]:
    for ch in ["，", "、", ";", "；", "\n", "\t"]:
        text = text.replace(ch, ",")
    tickers = []
    for x in text.split(","):
        x = normalize_stock_id(x.strip())
        if x and x not in tickers:
            tickers.append(x)
    return tickers


def merge_unique(items: Iterable[str]) -> list[str]:
    out = []
    for item in items:
        item = normalize_stock_id(item)
        if item and item not in out:
            out.append(item)
    return out


def stock_name_map(token: str) -> dict[str, str]:
    names = dict(KNOWN_NAMES)
    try:
        info = load_stock_info(token)
        if "stock_name" in info.columns:
            names.update(dict(zip(info["stock_id"].astype(str), info["stock_name"].astype(str))))
    except Exception:
        pass
    return names



def parse_uploaded_watchlist(uploaded_file) -> list[str]:
    """Read ticker symbols from a small CSV/TXT file without persisting it on the server."""
    if uploaded_file is None:
        return []
    data = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()

    if name.endswith(".csv"):
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "big5"):
            try:
                table = pd.read_csv(BytesIO(data), encoding=encoding, dtype=str)
                if table.empty:
                    return []
                preferred = [c for c in ["代號", "stock_id", "ticker", "symbol", "股票代號"] if c in table.columns]
                col = preferred[0] if preferred else table.columns[0]
                return merge_unique(table[col].astype(str).tolist())
            except Exception as exc:
                last_error = exc
        raise ValueError(f"CSV 無法讀取：{last_error}")

    decoded = None
    for encoding in ("utf-8-sig", "utf-8", "big5"):
        try:
            decoded = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ValueError("文字檔編碼無法辨識，請改存成 UTF-8 CSV/TXT。")
    return parse_tickers(decoded)


def watchlist_csv_bytes(tickers: list[str], names: dict[str, str]) -> bytes:
    table = pd.DataFrame({
        "代號": tickers,
        "名稱": [names.get(t, KNOWN_NAMES.get(t, "")) for t in tickers],
    })
    return table.to_csv(index=False).encode("utf-8-sig")


if "watchlist_text" not in st.session_state:
    st.session_state["watchlist_text"] = ", ".join(WATCHLISTS["大型權值"][:10])

secret_finmind_token = get_streamlit_secret("FINMIND_TOKEN")

st.title("📈 免費台股 AI 多空分析系統 v8")
st.caption("投資組合模擬器｜通知中心｜前向驗證｜每日自動快照｜策略研究實驗室｜Walk-forward")

with st.sidebar:
    st.header("個股分析設定")
    with st.form("query_form"):
        ticker_input = st.text_input("股票代號", value=st.session_state.get("ticker", "2330"), help="例如：2330、2317、2454")
        years = st.select_slider("歷史資料期間", options=[3, 5, 8], value=st.session_state.get("years", 3))
        source_options = ["自動（FinMind → Yahoo 備援）", "只用 FinMind", "只用 Yahoo Finance"]
        saved_source = st.session_state.get("source_label", source_options[0])
        saved_index = source_options.index(saved_source) if saved_source in source_options else 0
        source_label = st.selectbox("資料來源", source_options, index=saved_index)
        token = st.text_input("FinMind Token（選填）", value=st.session_state.get("finmind_token", secret_finmind_token), type="password",
                              help="不填可用匿名免費額度；token 可提高 FinMind 額度。")
        threshold = st.slider("AI 持有門檻", min_value=0.50, max_value=0.70,
                              value=float(st.session_state.get("threshold", 0.55)), step=0.01,
                              help="回測時，未來 5 日上漲機率達此門檻才持有。")
        retrain_every = st.select_slider("Walk-forward 重訓頻率（交易日）", options=[10, 20, 30, 40, 60],
                                         value=int(st.session_state.get("retrain_every", 30)))
        one_way_cost_pct = st.number_input("回測單邊成本假設（%）", min_value=0.0, max_value=1.0,
                                           value=float(st.session_state.get("one_way_cost_pct", 0.15)), step=0.05,
                                           help="只是回測假設，不代表特定券商實際費率。")
        submitted = st.form_submit_button("開始分析", use_container_width=True)

    if submitted or "ticker" not in st.session_state:
        st.session_state["ticker"] = ticker_input.strip()
        st.session_state["years"] = years
        st.session_state["source_label"] = source_label
        st.session_state["finmind_token"] = token
        st.session_state["threshold"] = threshold
        st.session_state["retrain_every"] = retrain_every
        st.session_state["one_way_cost_pct"] = one_way_cost_pct

    st.divider()
    st.caption("v8 為研究工具，不是投資建議。『AI』是歷史價格/成交量的機器學習機率，不是保證預測。")

stock_input = st.session_state.get("ticker", "2330")
years = int(st.session_state.get("years", 3))
source_label = st.session_state.get("source_label", "自動（FinMind → Yahoo 備援）")
token = st.session_state.get("finmind_token", "")
threshold = float(st.session_state.get("threshold", 0.55))
retrain_every = int(st.session_state.get("retrain_every", 30))
one_way_cost = float(st.session_state.get("one_way_cost_pct", 0.15)) / 100
source_map = {
    "自動（FinMind → Yahoo 備援）": "auto",
    "只用 FinMind": "finmind",
    "只用 Yahoo Finance": "yahoo",
}
source = source_map.get(source_label, "auto")
stock_id = normalize_stock_id(stock_input)
names = stock_name_map(token)
stock_name = names.get(stock_id, "")

try:
    with st.spinner(f"正在取得 {stock_id} 的資料並執行 walk-forward…"):
        raw_df, used_source = load_data(stock_input, years, source, token)
        df, model = prepare_data(raw_df, threshold, retrain_every)
except Exception as exc:
    st.error(f"目前無法完成個股分析：{exc}")
    st.info("可以先確認股票代號，或把資料來源改成『自動』；若 FinMind 額度用完，系統會嘗試 Yahoo Finance。")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2]
support, resistance = recent_support_resistance(df, 20)
score = float(latest["strength_score"])
prob = model.probability_up_5d
latest_signal = latest["signal"] or "—"
trend = strength_label(score)
price_delta = float(latest["close"] / prev["close"] - 1)

def robustness_heatmap_chart(pivot: pd.DataFrame, title: str, percent: bool = True) -> go.Figure:
    if pivot.empty:
        return go.Figure()
    z = pivot.values.astype(float)
    text = np.empty_like(z, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            text[i, j] = "—" if not np.isfinite(v) else (f"{v*100:.1f}%" if percent else f"{v:.2f}")
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[str(x) for x in pivot.columns], y=[str(y) for y in pivot.index],
        text=text, texttemplate="%{text}", hovertemplate="最低力道=%{x}<br>最低量比=%{y}<br>%{text}<extra></extra>",
        colorbar=dict(title="值"),
    ))
    fig.update_layout(title=title, xaxis_title="最低力道", yaxis_title="最低量比", height=430)
    return fig


header_name = f" {stock_name}" if stock_name else ""
st.subheader(f"{stock_id}{header_name}｜最新分析：{latest['date'].date()}")
st.caption(f"資料來源：{used_source}｜共 {len(df):,} 個交易日｜模型採 expanding-window walk-forward")

c1, c2, c3, c4 = st.columns(4)
c1.metric("收盤價", f"{latest['close']:.2f}", pct_text(price_delta))
c2.metric("四色多空力道", f"{score:.0f}/100", trend)
c3.metric("未來 5 日上漲機率", f"{prob * 100:.1f}%", "機器學習估計")
c4.metric("最新 V/A", latest_signal, "V=確認轉強、A=確認轉弱")

s1, s2, s3, s4 = st.columns(4)
s1.metric("20 日支撐參考", f"{support:.2f}")
s2.metric("20 日壓力參考", f"{resistance:.2f}")
s3.metric("RSI(14)", f"{latest['rsi14']:.1f}")
s4.metric("成交量 / 20日均量", f"{latest['volume_ratio']:.2f}x")

if prob >= 0.60 and score >= 60:
    st.success("技術力道與模型機率目前同向偏多。這是研究訊號，不是買進建議。")
elif prob <= 0.40 and score <= 40:
    st.warning("技術力道與模型機率目前同向偏空。這是研究訊號，不是賣出建議。")
else:
    st.info("技術力道與模型機率目前沒有形成強烈同向，可視為中性或分歧。")

summary_tab, strength_tab, signal_tab, strategy_tab, portfolio_tab, robustness_tab, model_tab, daily_tab, screener_tab, forward_tab, notification_tab, data_tab = st.tabs(
    ["個股總覽", "四色力道 K + V/A", "V/A 歷史統計", "策略研究實驗室", "投資組合模擬", "參數穩健性", "AI / Walk-forward", "每日訊號中心", "觀察池選股排行", "前向驗證", "通知中心", "資料"]
)

with summary_tab:
    st.plotly_chart(price_chart(df), use_container_width=True)
    st.markdown("#### 技術指標")
    st.plotly_chart(indicator_chart(df), use_container_width=True)
    table = df[["date", "close", "ma20", "ma60", "rsi14", "macd_hist", "volume_ratio", "strength_score", "signal"]].tail(15).copy()
    table = table.sort_values("date", ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)

with strength_tab:
    st.plotly_chart(strength_candle_chart(df), use_container_width=True)
    st.caption("v8 沿用四色規則：強多(≥70)、偏多(50–69)、偏空(30–49)、強空(<30)。V/A 會要求 MA20 與 MACD 同向確認。")
    st.caption("四色力道、翻轉線與 V/A 規則都是本專案自行設計，不是參考網站的專有公式。")


with signal_tab:
    st.markdown("#### V/A 歷史訊號統計")
    st.caption("這裡是事後檢驗：V 訊號後價格上漲算成功；A 訊號後價格下跌算成功。未來價格只用來評估，不會參與產生訊號。")
    stats = signal_performance_stats(df, horizons=(5, 10, 20))
    show_stats = stats.copy()
    show_stats["勝率"] = (show_stats["勝率"] * 100).round(1)
    show_stats["平均方向報酬"] = (show_stats["平均方向報酬"] * 100).round(2)
    show_stats["中位方向報酬"] = (show_stats["中位方向報酬"] * 100).round(2)
    show_stats["平均原始漲跌"] = (show_stats["平均原始漲跌"] * 100).round(2)
    show_stats = show_stats.rename(columns={
        "勝率": "勝率%",
        "平均方向報酬": "平均方向報酬%",
        "中位方向報酬": "中位方向報酬%",
        "平均原始漲跌": "平均原始漲跌%",
    })
    st.dataframe(show_stats, use_container_width=True, hide_index=True)

    five = stats[stats["期間"] == "5日"].set_index("訊號")
    v_row = five.loc["V"] if "V" in five.index else None
    a_row = five.loc["A"] if "A" in five.index else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("V 後 5 日勝率", "—" if v_row is None or pd.isna(v_row["勝率"]) else f"{v_row['勝率'] * 100:.1f}%")
    c2.metric("V 樣本數", "0" if v_row is None else f"{int(v_row['樣本數'])}")
    c3.metric("A 後 5 日勝率", "—" if a_row is None or pd.isna(a_row["勝率"]) else f"{a_row['勝率'] * 100:.1f}%")
    c4.metric("A 樣本數", "0" if a_row is None else f"{int(a_row['樣本數'])}")

    st.markdown("#### 期望報酬 / 盈虧比")
    econ = signal_expectancy_stats(df, horizons=(5, 10, 20))
    econ_show = econ.copy()
    for c in ["勝率", "期望方向報酬", "平均獲利", "平均虧損"]:
        econ_show[c] = (pd.to_numeric(econ_show[c], errors="coerce") * 100).round(2)
    for c in ["盈虧比", "Profit Factor"]:
        econ_show[c] = pd.to_numeric(econ_show[c], errors="coerce").round(2)
    econ_show = econ_show.rename(columns={
        "勝率": "勝率%",
        "期望方向報酬": "期望方向報酬%",
        "平均獲利": "平均獲利%",
        "平均虧損": "平均虧損%",
    })
    st.dataframe(econ_show, use_container_width=True, hide_index=True)

    econ5 = econ[econ["期間"] == "5日"].set_index("訊號")
    ev = econ5.loc["V"] if "V" in econ5.index else None
    ea = econ5.loc["A"] if "A" in econ5.index else None
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("V 5日期望方向報酬", "—" if ev is None or pd.isna(ev["期望方向報酬"]) else f"{ev['期望方向報酬'] * 100:.2f}%")
    e2.metric("V 5日盈虧比", "—" if ev is None or pd.isna(ev["盈虧比"]) else f"{ev['盈虧比']:.2f}")
    e3.metric("A 5日期望方向報酬", "—" if ea is None or pd.isna(ea["期望方向報酬"]) else f"{ea['期望方向報酬'] * 100:.2f}%")
    e4.metric("A 5日盈虧比", "—" if ea is None or pd.isna(ea["盈虧比"]) else f"{ea['盈虧比']:.2f}")

    recent_v_idxs = df.index[df["signal"].eq("V")].tolist()
    if recent_v_idxs:
        last_v_idx = int(recent_v_idxs[-1])
        q = v_signal_quality(df, signal_index=last_v_idx, horizon=5)
        v_date = pd.Timestamp(df.loc[last_v_idx, "date"]).date().isoformat()
        bars_after_v = len(df) - 1 - last_v_idx
        st.markdown("#### 最近一次 V 的研究品質")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("最近 V", v_date, f"距今 {bars_after_v} 個交易日")
        q2.metric("V 品質等級", str(q["等級"]), "A → D 僅為研究分級")
        q3.metric("品質分數", "—" if pd.isna(q["分數"]) else f"{q['分數']:.1f}/100")
        q4.metric("當時可用歷史樣本", f"{int(q['歷史樣本'])}")
        st.caption("V 品質分數：70% 來自訊號當下的力道、趨勢、MACD、RSI、量能與區間位置；30% 最多來自該 V 以前已完成的歷史 V 樣本。它不會使用這次 V 之後才發生的股價。")

    st.markdown("#### 最近 V/A 紀錄")
    log_horizon = st.selectbox("檢查訊號後幾個交易日", [5, 10, 20], index=0, key="signal_log_horizon")
    log = recent_signal_log(df, horizon=int(log_horizon), limit=30)
    if log.empty:
        st.info("目前歷史區間內沒有 V/A 訊號。")
    else:
        for col in ["當日收盤", "力道", f"{log_horizon}日後漲跌%"]:
            log[col] = pd.to_numeric(log[col], errors="coerce").round(2)
        st.dataframe(log, use_container_width=True, hide_index=True)
        st.download_button(
            "下載 V/A 歷史紀錄 CSV",
            log.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{stock_id}_va_history_v8.csv",
            mime="text/csv",
        )
    st.caption("樣本數少時勝率容易大幅波動；請同時看樣本數、不同期間與 walk-forward 結果，不要只看單一百分比。")


with strategy_tab:
    st.markdown("#### V 訊號策略研究實驗室")
    st.caption(
        "把 V 訊號轉成可重複檢驗的規則。訊號在收盤後才知道，因此一律使用『下一交易日開盤』進場；"
        "同一時間只持有一筆部位，避免把同一筆資金重複計算。"
    )

    with st.form("strategy_lab_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            grade_options = st.multiselect(
                "允許的 V 品質",
                ["A", "B", "C", "D"],
                default=["A", "B"],
                help="品質分級只使用訊號當下資料與更早已完成的 V 樣本。",
            )
            lab_min_strength = st.slider("最低力道", 50, 90, 55, 1)
            lab_min_volume = st.slider("最低量比", 0.5, 3.0, 1.0, 0.1)
        with p2:
            momentum_enabled = st.checkbox("啟用 20 日動能門檻", value=False)
            lab_min_momentum = st.slider("最低 20 日動能（%）", -20.0, 40.0, 0.0, 1.0, disabled=not momentum_enabled)
            lab_hold_days = st.selectbox("最長持有交易日", [5, 10, 20], index=1)
        with p3:
            stop_enabled = st.checkbox("啟用停損", value=True)
            lab_stop = st.slider("停損（%）", 1.0, 20.0, 6.0, 0.5, disabled=not stop_enabled)
            take_enabled = st.checkbox("啟用停利", value=True)
            lab_take = st.slider("停利（%）", 2.0, 40.0, 12.0, 0.5, disabled=not take_enabled)

        run_lab = st.form_submit_button("🧪 執行策略回測", type="primary", use_container_width=True)

    if run_lab:
        try:
            trades, equity, lab_stats = strategy_lab_backtest(
                df,
                allowed_grades=tuple(grade_options),
                min_strength=float(lab_min_strength),
                min_volume_ratio=float(lab_min_volume),
                min_momentum_20=float(lab_min_momentum) if momentum_enabled else None,
                hold_days=int(lab_hold_days),
                stop_loss_pct=float(lab_stop) / 100 if stop_enabled else None,
                take_profit_pct=float(lab_take) / 100 if take_enabled else None,
                one_way_cost=float(one_way_cost),
            )
            st.session_state["strategy_lab_v6"] = {
                "trades": trades,
                "equity": equity,
                "stats": lab_stats,
                "stock_id": stock_id,
                "data_len": len(df),
                "data_last_date": pd.Timestamp(df.iloc[-1]["date"]).date().isoformat(),
                "params": {
                    "grades": grade_options,
                    "min_strength": lab_min_strength,
                    "min_volume": lab_min_volume,
                    "momentum_enabled": momentum_enabled,
                    "min_momentum": lab_min_momentum,
                    "hold_days": lab_hold_days,
                    "stop_enabled": stop_enabled,
                    "stop": lab_stop,
                    "take_enabled": take_enabled,
                    "take": lab_take,
                    "one_way_cost": one_way_cost,
                },
            }
        except Exception as exc:
            st.error(f"策略回測無法完成：{exc}")

    lab = st.session_state.get("strategy_lab_v6")
    current_data_last_date = pd.Timestamp(df.iloc[-1]["date"]).date().isoformat()
    lab_matches_current_data = (
        isinstance(lab, dict)
        and lab.get("stock_id") == stock_id
        and lab.get("data_len") == len(df)
        and lab.get("data_last_date") == current_data_last_date
    )
    if lab_matches_current_data:
        trades = lab.get("trades", pd.DataFrame())
        equity = lab.get("equity", pd.DataFrame())
        lab_stats = lab.get("stats", {})
        params = lab.get("params", {})

        if isinstance(trades, pd.DataFrame) and trades.empty:
            st.warning(
                "這組條件沒有產生實際交易。可以嘗試加入 C 級、降低最低力道或量比，或把歷史資料期間改成 5/8 年。"
            )
            q1, q2 = st.columns(2)
            q1.metric("符合條件 V 訊號", int(lab_stats.get("qualifying_signals", 0)))
            q2.metric("實際交易", 0)
        elif isinstance(trades, pd.DataFrame):
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("實際交易", f"{int(lab_stats.get('trades', 0))}")
            r2.metric("勝率", pct_text(lab_stats.get("win_rate", np.nan)))
            r3.metric("每筆期望淨報酬", pct_text(lab_stats.get("expectancy", np.nan), 2))
            r4.metric("盈虧比", "—" if pd.isna(lab_stats.get("payoff_ratio", np.nan)) else f"{lab_stats['payoff_ratio']:.2f}")
            r5.metric("Profit Factor", "—" if pd.isna(lab_stats.get("profit_factor", np.nan)) else f"{lab_stats['profit_factor']:.2f}")

            r6, r7, r8, r9, r10 = st.columns(5)
            r6.metric("累積報酬", pct_text(lab_stats.get("total_return", np.nan)))
            r7.metric("最大回撤", pct_text(lab_stats.get("max_drawdown", np.nan)))
            r8.metric("平均獲利", pct_text(lab_stats.get("avg_win", np.nan), 2))
            r9.metric("平均虧損", pct_text(lab_stats.get("avg_loss", np.nan), 2))
            r10.metric("平均持有日", f"{lab_stats.get('avg_holding_days', np.nan):.1f}")

            st.plotly_chart(strategy_equity_chart(equity), use_container_width=True)

            qualifying = int(lab_stats.get("qualifying_signals", 0))
            overlap = int(lab_stats.get("skipped_overlap", 0))
            incomplete = int(lab_stats.get("skipped_incomplete", 0))
            st.caption(
                f"符合篩選的 V 訊號 {qualifying} 次；其中 {overlap} 次因前一筆交易尚未出場而略過，"
                f"{incomplete} 次因最新資料尚未走滿持有期而不納入績效。"
                f"交易成本：單邊 {float(params.get('one_way_cost', one_way_cost)):.2%}。"
            )

            st.markdown("##### 逐筆交易紀錄")
            trade_show = trades.copy()
            for c in ["毛報酬", "淨報酬"]:
                trade_show[c.replace("報酬", "報酬%") ] = (pd.to_numeric(trade_show[c], errors="coerce") * 100).round(2)
            trade_show = trade_show.drop(columns=[c for c in ["毛報酬", "淨報酬"] if c in trade_show.columns])
            for c in ["V品質分數", "訊號力道", "訊號量比", "訊號20日動能%", "進場價", "出場價"]:
                if c in trade_show.columns:
                    trade_show[c] = pd.to_numeric(trade_show[c], errors="coerce").round(2)
            st.dataframe(trade_show.sort_values("進場日", ascending=False), use_container_width=True, hide_index=True)
            st.download_button(
                "下載策略交易紀錄 CSV",
                trade_show.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{stock_id}_strategy_lab_v7.csv",
                mime="text/csv",
            )

            st.markdown("##### 相同條件：持有 5 / 10 / 20 日快速比較")
            compare_rows = []
            for compare_hold in (5, 10, 20):
                _, _, cs = strategy_lab_backtest(
                    df,
                    allowed_grades=tuple(params.get("grades", ["A", "B"])),
                    min_strength=float(params.get("min_strength", 55)),
                    min_volume_ratio=float(params.get("min_volume", 1.0)),
                    min_momentum_20=float(params.get("min_momentum", 0.0)) if params.get("momentum_enabled") else None,
                    hold_days=compare_hold,
                    stop_loss_pct=float(params.get("stop", 6.0)) / 100 if params.get("stop_enabled") else None,
                    take_profit_pct=float(params.get("take", 12.0)) / 100 if params.get("take_enabled") else None,
                    one_way_cost=float(params.get("one_way_cost", one_way_cost)),
                )
                compare_rows.append({
                    "持有上限": f"{compare_hold}日",
                    "交易數": int(cs.get("trades", 0)),
                    "勝率%": cs.get("win_rate", np.nan) * 100,
                    "期望淨報酬%": cs.get("expectancy", np.nan) * 100,
                    "盈虧比": cs.get("payoff_ratio", np.nan),
                    "Profit Factor": cs.get("profit_factor", np.nan),
                    "累積報酬%": cs.get("total_return", np.nan) * 100,
                    "最大回撤%": cs.get("max_drawdown", np.nan) * 100,
                })
            compare_df = pd.DataFrame(compare_rows)
            for c in ["勝率%", "期望淨報酬%", "盈虧比", "Profit Factor", "累積報酬%", "最大回撤%"]:
                compare_df[c] = pd.to_numeric(compare_df[c], errors="coerce").round(2)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

    with st.expander("v7 策略回測規則與限制"):
        st.markdown(
            "- **進場**：V 訊號當天收盤後才知道，因此下一交易日開盤進場，避免偷看未來。\n"
            "- **篩選**：品質、力道、量比、20 日動能都只使用 V 訊號當天已知資料。\n"
            "- **部位**：一次只持有一筆、假設每筆使用全部研究資金；持倉中出現的新 V 會略過。\n"
            "- **停損 / 停利**：用日 K 高低價判斷；若同一天兩者都被觸及，採較保守的『停損先發生』假設。\n"
            "- **跳空**：若開盤已越過停損/停利價，使用實際開盤價出場。\n"
            "- **成本**：進場與出場各扣一次側欄設定的單邊成本。尚未模擬滑價、股利與完整稅費差異。"
        )



with portfolio_tab:
    st.markdown("#### 💼 v8 投資組合模擬器")
    st.caption(
        "把多檔股票的 V 訊號放進同一筆資金裡研究。V 在收盤後才確認，因此仍使用下一交易日開盤進場；"
        "同日候選過多時，依 V 品質、品質分數、力道與量比排序。模擬不使用槓桿。"
    )

    if "portfolio_universe_v8" not in st.session_state:
        st.session_state["portfolio_universe_v8"] = ", ".join(WATCHLISTS["大型權值"][:8])

    pool_options = ["大型權值", "AI / 電子", "金融", "ETF", "自訂"]
    ptop1, ptop2 = st.columns([1, 2])
    with ptop1:
        portfolio_pool = st.selectbox("股票池", pool_options, key="portfolio_pool_v8")
    with ptop2:
        if portfolio_pool != "自訂":
            suggested = ", ".join(WATCHLISTS[portfolio_pool][:12])
            if st.button("套用這個股票池", key="apply_portfolio_pool_v8"):
                st.session_state["portfolio_universe_v8"] = suggested
                st.rerun()
        portfolio_text = st.text_area(
            "股票代號（逗號分隔，免費版建議 5～12 檔）",
            key="portfolio_universe_v8",
            height=82,
            help="例如：2330, 2317, 2454, 2308, 2382。一次太多股票會增加免費 API 呼叫與運算時間。",
        )

    with st.form("portfolio_lab_form_v8"):
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            pf_initial = st.number_input("初始資金（元）", min_value=100_000, max_value=100_000_000, value=1_000_000, step=100_000)
            pf_max_positions = st.slider("最多同時持股", 1, 10, 5, 1)
            pf_position_pct = st.slider("單檔目標資金比例（%）", 5, 50, 20, 5)
        with r2:
            pf_grades = st.multiselect("允許 V 品質", ["A", "B", "C", "D"], default=["A", "B"])
            pf_min_strength = st.slider("最低力道", 50, 90, 55, 1, key="pf_min_strength_v8")
            pf_min_volume = st.slider("最低量比", 0.5, 3.0, 1.0, 0.1, key="pf_min_volume_v8")
        with r3:
            pf_momentum_enabled = st.checkbox("啟用 20 日動能門檻", value=False, key="pf_momentum_enabled_v8")
            pf_min_momentum = st.slider("最低 20 日動能（%）", -20.0, 40.0, 0.0, 1.0, disabled=not pf_momentum_enabled, key="pf_min_momentum_v8")
            pf_hold_days = st.selectbox("最長持有交易日", [5, 10, 20], index=1, key="pf_hold_days_v8")
        with r4:
            pf_stop_enabled = st.checkbox("啟用停損", value=True, key="pf_stop_enabled_v8")
            pf_stop = st.slider("停損（%）", 1.0, 20.0, 6.0, 0.5, disabled=not pf_stop_enabled, key="pf_stop_v8")
            pf_take_enabled = st.checkbox("啟用停利", value=True, key="pf_take_enabled_v8")
            pf_take = st.slider("停利（%）", 2.0, 40.0, 12.0, 0.5, disabled=not pf_take_enabled, key="pf_take_v8")

        compare_0050 = st.checkbox("與 0050 買進持有做歷史基準比較", value=True)
        run_portfolio = st.form_submit_button("💼 執行投資組合模擬", type="primary", use_container_width=True)

    if pf_position_pct * pf_max_positions > 100:
        st.info("單檔比例 × 最大持股數超過 100%。v8 不會使用槓桿；現金不足時會自動縮小或略過後續進場。")

    if run_portfolio:
        tickers = parse_tickers(portfolio_text)
        if not tickers:
            st.error("請至少輸入一個股票代號。")
        else:
            if len(tickers) > 12:
                st.warning("為了免費 API 與運算穩定，v8 一次最多模擬前 12 檔。")
                tickers = tickers[:12]

            data_map = {}
            failures = []
            progress = st.progress(0.0, text="正在取得投資組合歷史資料…")
            for i, sid in enumerate(tickers, start=1):
                try:
                    raw_pf, _ = load_data(sid, years, source, token)
                    data_map[sid] = add_indicators(raw_pf)
                except Exception as exc:
                    failures.append(f"{sid}: {exc}")
                progress.progress(i / max(len(tickers), 1), text=f"正在處理 {sid}（{i}/{len(tickers)}）")
            progress.empty()

            if failures:
                st.warning("部分股票無法取得資料，已略過：" + "；".join(failures[:5]))

            if not data_map:
                st.error("這次沒有任何股票可供模擬，請檢查代號或資料來源。")
            else:
                try:
                    trades_pf, equity_pf, stats_pf, contribution_pf = portfolio_backtest(
                        data_map,
                        stock_names=names,
                        initial_capital=float(pf_initial),
                        allowed_grades=tuple(pf_grades),
                        min_strength=float(pf_min_strength),
                        min_volume_ratio=float(pf_min_volume),
                        min_momentum_20=float(pf_min_momentum) if pf_momentum_enabled else None,
                        hold_days=int(pf_hold_days),
                        stop_loss_pct=float(pf_stop) / 100 if pf_stop_enabled else None,
                        take_profit_pct=float(pf_take) / 100 if pf_take_enabled else None,
                        one_way_cost=float(one_way_cost),
                        max_positions=int(pf_max_positions),
                        position_pct=float(pf_position_pct) / 100,
                    )

                    benchmark_pf = pd.DataFrame()
                    benchmark_stats_pf = {}
                    if compare_0050 and not equity_pf.empty:
                        try:
                            if "0050" in data_map:
                                bench_raw = data_map["0050"]
                            else:
                                bench_raw, _ = load_data("0050", years, source, token)
                            benchmark_pf, benchmark_stats_pf = build_benchmark_curve(
                                bench_raw,
                                equity_pf["date"],
                                float(pf_initial),
                            )
                        except Exception as exc:
                            st.warning(f"0050 基準暫時無法建立：{exc}")

                    st.session_state["portfolio_result_v8"] = {
                        "trades": trades_pf,
                        "equity": equity_pf,
                        "stats": stats_pf,
                        "contribution": contribution_pf,
                        "benchmark": benchmark_pf,
                        "benchmark_stats": benchmark_stats_pf,
                        "tickers": list(data_map.keys()),
                        "years": years,
                        "latest_dates": {k: pd.Timestamp(v["date"].max()).date().isoformat() for k, v in data_map.items()},
                        "params": {
                            "initial": pf_initial,
                            "max_positions": pf_max_positions,
                            "position_pct": pf_position_pct,
                            "grades": pf_grades,
                            "min_strength": pf_min_strength,
                            "min_volume": pf_min_volume,
                            "hold_days": pf_hold_days,
                            "stop": pf_stop if pf_stop_enabled else None,
                            "take": pf_take if pf_take_enabled else None,
                        },
                    }
                except Exception as exc:
                    st.error(f"投資組合模擬無法完成：{exc}")

    pf = st.session_state.get("portfolio_result_v8")
    if isinstance(pf, dict):
        trades_pf = pf.get("trades", pd.DataFrame())
        equity_pf = pf.get("equity", pd.DataFrame())
        stats_pf = pf.get("stats", {})
        contribution_pf = pf.get("contribution", pd.DataFrame())
        benchmark_pf = pf.get("benchmark", pd.DataFrame())
        benchmark_stats_pf = pf.get("benchmark_stats", {})
        params_pf = pf.get("params", {})

        st.markdown("#### 模擬結果")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("期末模擬資產", f"${stats_pf.get('ending_equity', 0):,.0f}")
        m2.metric("累積報酬", pct_text(stats_pf.get("total_return"), 1))
        m3.metric("年化報酬", pct_text(stats_pf.get("cagr"), 1))
        m4.metric("最大回撤", pct_text(stats_pf.get("max_drawdown"), 1))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Sharpe（0%無風險利率）", "—" if pd.isna(stats_pf.get("sharpe")) else f"{stats_pf.get('sharpe'):.2f}")
        m6.metric("完成交易", f"{int(stats_pf.get('trades', 0))}")
        m7.metric("交易勝率", pct_text(stats_pf.get("win_rate"), 1))
        m8.metric("平均市場曝險", pct_text(stats_pf.get("exposure"), 1))

        if benchmark_stats_pf:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("0050 同期累積報酬", pct_text(benchmark_stats_pf.get("total_return"), 1))
            b2.metric("0050 同期年化報酬", pct_text(benchmark_stats_pf.get("cagr"), 1))
            b3.metric("0050 同期最大回撤", pct_text(benchmark_stats_pf.get("max_drawdown"), 1))
            if pd.notna(stats_pf.get("total_return")) and pd.notna(benchmark_stats_pf.get("total_return")):
                excess = float(stats_pf.get("total_return")) - float(benchmark_stats_pf.get("total_return"))
                b4.metric("歷史累積報酬差", f"{excess * 100:+.1f} 個百分點")
            else:
                b4.metric("歷史累積報酬差", "—")

        if not equity_pf.empty:
            st.plotly_chart(portfolio_equity_chart(equity_pf, benchmark_pf), use_container_width=True)
            st.plotly_chart(portfolio_position_chart(equity_pf), use_container_width=True)

        x1, x2, x3, x4 = st.columns(4)
        x1.metric("候選 V 訊號", f"{int(stats_pf.get('candidate_signals', 0))}")
        x2.metric("因滿倉略過", f"{int(stats_pf.get('skipped_slots', 0))}")
        x3.metric("因現金不足略過", f"{int(stats_pf.get('skipped_no_cash', 0))}")
        x4.metric("歷史最高同時持股", f"{int(stats_pf.get('max_positions_used', 0))}")

        st.markdown("#### 個股貢獻拆解")
        if isinstance(contribution_pf, pd.DataFrame) and not contribution_pf.empty:
            contrib_show = contribution_pf.copy()
            contrib_show["勝率"] = (pd.to_numeric(contrib_show["勝率"], errors="coerce") * 100).round(1)
            contrib_show["平均淨報酬"] = (pd.to_numeric(contrib_show["平均淨報酬"], errors="coerce") * 100).round(2)
            contrib_show["損益金額"] = pd.to_numeric(contrib_show["損益金額"], errors="coerce").round(0)
            contrib_show = contrib_show.rename(columns={"勝率": "勝率%", "平均淨報酬": "平均淨報酬%"})
            st.dataframe(contrib_show, use_container_width=True, hide_index=True)
        else:
            st.info("這組條件沒有完成交易，因此尚無個股貢獻。")

        st.markdown("#### 逐筆交易")
        if isinstance(trades_pf, pd.DataFrame) and not trades_pf.empty:
            show_trades = trades_pf.copy().sort_values("進場日", ascending=False)
            for c in ["毛報酬", "淨報酬"]:
                show_trades[c] = (pd.to_numeric(show_trades[c], errors="coerce") * 100).round(2)
            for c in ["進場價", "出場價", "投入金額", "損益金額", "V品質分數", "訊號力道", "訊號量比", "訊號20日動能%"]:
                show_trades[c] = pd.to_numeric(show_trades[c], errors="coerce").round(2)
            show_trades = show_trades.rename(columns={"毛報酬": "毛報酬%", "淨報酬": "淨報酬%"})
            st.dataframe(show_trades, use_container_width=True, hide_index=True)
            st.download_button(
                "下載投資組合逐筆交易 CSV",
                trades_pf.to_csv(index=False).encode("utf-8-sig"),
                file_name="portfolio_backtest_v8.csv",
                mime="text/csv",
            )
        else:
            st.warning("目前條件沒有完成交易。可以增加 C 級、降低力道/量比門檻，或改用較長歷史區間。")

        st.caption(
            "v8 是歷史模擬，不代表未來績效。0050 基準使用同期每日收盤價正規化；投資組合則包含你設定的交易成本、"
            "下一日開盤進場、停損/停利與資金/持股上限，因此兩者是研究比較，不是完全相同的交易執行假設。"
        )
        with st.expander("v8 如何避免常見回測作弊？"):
            st.markdown(
                "- V 訊號只在當日收盤後成立，最早下一交易日開盤進場。\n"
                "- V 品質不使用該次訊號未來才發生的價格。\n"
                "- 同日訊號過多時使用固定排序規則，不事後挑最會漲的股票。\n"
                "- 不使用槓桿；現金不足就縮小/略過進場。\n"
                "- 同一天同時碰到停損與停利，採保守的停損優先。\n"
                "- 當天才在收盤出場的持股，早上仍占用持股名額；不把收盤才拿回的資金假裝拿去早盤買新股票。"
            )


with robustness_tab:
    st.markdown("#### 🧭 v9 參數穩健性研究")
    st.caption(
        "一次掃描多組參數，重點不是找『歷史最高報酬』，而是觀察鄰近參數是否也能維持正期望、正報酬與可接受回撤。"
        "v9 使用與 v8 相同的投資組合回測引擎，並限制組合數，避免免費主機運算過重。"
    )

    if "robust_universe_v9" not in st.session_state:
        st.session_state["robust_universe_v9"] = ", ".join(WATCHLISTS["大型權值"][:6])

    rr1, rr2 = st.columns([1, 2])
    with rr1:
        robust_pool = st.selectbox("研究股票池", ["大型權值", "AI / 電子", "金融", "ETF", "自訂"], key="robust_pool_v9")
    with rr2:
        if robust_pool != "自訂" and st.button("套用研究股票池", key="apply_robust_pool_v9"):
            st.session_state["robust_universe_v9"] = ", ".join(WATCHLISTS[robust_pool][:8])
            st.rerun()
        robust_text = st.text_area(
            "股票代號（建議 4～8 檔）", key="robust_universe_v9", height=82,
            help="參數掃描會重複回測很多次，因此比一般 v8 模擬更吃運算；免費版建議先用 4～8 檔。",
        )

    with st.form("robustness_form_v9"):
        a1, a2, a3 = st.columns(3)
        with a1:
            rb_initial = st.number_input("初始資金（元）", min_value=100_000, max_value=100_000_000, value=1_000_000, step=100_000, key="rb_initial_v9")
            rb_grades = st.multiselect("允許 V 品質", ["A", "B", "C", "D"], default=["A", "B"], key="rb_grades_v9")
            rb_min_trades = st.slider("每組至少交易數", 3, 30, 8, 1, key="rb_min_trades_v9")
        with a2:
            rb_max_positions = st.slider("最多同時持股", 1, 10, 5, 1, key="rb_max_positions_v9")
            rb_position_pct = st.slider("單檔目標資金比例（%）", 5, 50, 20, 5, key="rb_position_pct_v9")
            rb_momentum_enabled = st.checkbox("啟用 20 日動能門檻", value=False, key="rb_momentum_enabled_v9")
            rb_min_momentum = st.slider("最低 20 日動能（%）", -20.0, 40.0, 0.0, 1.0, disabled=not rb_momentum_enabled, key="rb_min_momentum_v9")
        with a3:
            st.write("掃描上限：120 組")
            st.caption("若參數組合超過上限，請縮小其中一個範圍。")
            st.write(f"交易成本：單邊 {one_way_cost:.2%}")

        st.markdown("##### 掃描範圍")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            rb_strength = st.multiselect("最低力道", [50, 55, 60, 65, 70, 75], default=[55, 60, 65], key="rb_strength_v9")
        with b2:
            rb_volume = st.multiselect("最低量比", [0.8, 1.0, 1.2, 1.5, 2.0], default=[0.8, 1.0, 1.2], key="rb_volume_v9")
        with b3:
            rb_hold = st.multiselect("持有日", [5, 10, 20], default=[5, 10, 20], key="rb_hold_v9")
        with b4:
            rb_stop = st.multiselect("停損%", [0, 4, 6, 8, 10], default=[4, 6], key="rb_stop_v9", help="0 代表不設停損")
        with b5:
            rb_take = st.multiselect("停利%", [0, 8, 12, 16, 20], default=[8, 12], key="rb_take_v9", help="0 代表不設停利")

        combos = combination_count(rb_strength, rb_volume, rb_hold, rb_stop, rb_take)
        st.caption(f"目前共 {combos} 組參數。")
        run_robust = st.form_submit_button("🧭 執行穩健性掃描", type="primary", use_container_width=True, disabled=(combos == 0 or combos > 120))

    if combos > 120:
        st.warning(f"目前有 {combos} 組，超過免費版上限 120 組。請減少一個或多個參數值。")

    if run_robust:
        tickers = parse_tickers(robust_text)
        if not tickers:
            st.error("請至少輸入一個股票代號。")
        else:
            if len(tickers) > 8:
                st.warning("v9 免費版穩健性掃描一次最多使用前 8 檔。")
                tickers = tickers[:8]
            data_map_rb = {}
            failures_rb = []
            load_progress = st.progress(0.0, text="正在取得穩健性研究資料…")
            for i, sid in enumerate(tickers, start=1):
                try:
                    raw_rb, _ = load_data(sid, years, source, token)
                    data_map_rb[sid] = add_indicators(raw_rb)
                except Exception as exc:
                    failures_rb.append(f"{sid}: {exc}")
                load_progress.progress(i / max(len(tickers), 1), text=f"載入 {sid}（{i}/{len(tickers)}）")
            load_progress.empty()
            if failures_rb:
                st.warning("部分股票無法取得，已略過：" + "；".join(failures_rb[:5]))
            if not data_map_rb:
                st.error("沒有可用股票資料。")
            else:
                scan_progress = st.progress(0.0, text="正在掃描參數…")
                status_box = st.empty()
                def _progress(done, total, row):
                    scan_progress.progress(done / max(total, 1), text=f"參數掃描 {done}/{total}")
                    if done == 1 or done == total or done % 10 == 0:
                        status_box.caption(f"已完成 {done}/{total} 組；目前交易數 {int(row.get('trades', 0))}。")
                try:
                    rb_results, rb_meta = run_parameter_scan(
                        data_map_rb,
                        stock_names=names,
                        initial_capital=float(rb_initial),
                        allowed_grades=tuple(rb_grades),
                        strength_values=rb_strength,
                        volume_values=rb_volume,
                        hold_values=rb_hold,
                        stop_values=rb_stop,
                        take_values=rb_take,
                        min_momentum_20=float(rb_min_momentum) if rb_momentum_enabled else None,
                        one_way_cost=float(one_way_cost),
                        max_positions=int(rb_max_positions),
                        position_pct=float(rb_position_pct) / 100,
                        min_trades=int(rb_min_trades),
                        max_combinations=120,
                        progress_callback=_progress,
                    )
                    st.session_state["robustness_result_v9"] = {
                        "results": rb_results,
                        "meta": rb_meta,
                        "tickers": list(data_map_rb.keys()),
                        "years": years,
                        "min_trades": rb_min_trades,
                    }
                except Exception as exc:
                    st.error(f"穩健性掃描失敗：{exc}")
                finally:
                    scan_progress.empty()
                    status_box.empty()

    rb = st.session_state.get("robustness_result_v9")
    if isinstance(rb, dict) and isinstance(rb.get("results"), pd.DataFrame):
        results_rb = rb["results"].copy()
        meta_rb = rb.get("meta", {})
        min_trades_rb = int(rb.get("min_trades", 8))
        st.markdown("#### 掃描總覽")
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("參數組合", f"{int(meta_rb.get('combinations', 0))}")
        q2.metric("達最低樣本", f"{int(meta_rb.get('eligible_combinations', 0))}")
        q3.metric("穩健區組合", f"{int(meta_rb.get('robust_combinations', 0))}")
        q4.metric("正期望比例", pct_text(meta_rb.get("positive_expectancy_ratio"), 1))
        q5.metric("正報酬比例", pct_text(meta_rb.get("positive_return_ratio"), 1))

        eligible_rb = results_rb[pd.to_numeric(results_rb["trades"], errors="coerce") >= min_trades_rb].copy()
        if eligible_rb.empty:
            st.warning("沒有任何參數組合達到最低交易樣本數。可增加歷史年數、增加 C 級，或降低最低交易數。")
        else:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("年化報酬中位數", pct_text(meta_rb.get("median_cagr"), 1))
            r2.metric("每筆期望中位數", pct_text(meta_rb.get("median_expectancy"), 2))
            r3.metric("最大回撤中位數", pct_text(meta_rb.get("median_max_drawdown"), 1))
            robust_ratio = float(pd.to_numeric(eligible_rb["robust_region"], errors="coerce").fillna(0).mean())
            r4.metric("穩健區占比", pct_text(robust_ratio, 1))

            st.markdown("##### 穩健區候選（不是『最佳推薦』）")
            robust_rows = eligible_rb[eligible_rb["robust_region"].eq(True)].copy()
            if robust_rows.empty:
                st.info("目前沒有符合 v9 穩健區條件的組合。這本身就是研究結果：代表這個股票池/期間/訊號條件對參數很敏感。")
                robust_rows = eligible_rb.copy()
            robust_rows = robust_rows.sort_values(["robustness_score", "neighbor_median_expectancy", "neighbor_median_cagr"], ascending=False).head(30)
            show_rb = robust_rows[[
                "min_strength", "min_volume_ratio", "hold_days", "stop_loss_pct", "take_profit_pct", "trades",
                "total_return", "cagr", "max_drawdown", "sharpe", "expectancy", "win_rate", "robustness_score",
                "neighbor_positive_expectancy_ratio", "neighbor_positive_return_ratio", "isolated_peak"
            ]].copy()
            for c in ["total_return", "cagr", "max_drawdown", "expectancy", "win_rate", "neighbor_positive_expectancy_ratio", "neighbor_positive_return_ratio"]:
                show_rb[c] = (pd.to_numeric(show_rb[c], errors="coerce") * 100).round(2)
            show_rb["robustness_score"] = pd.to_numeric(show_rb["robustness_score"], errors="coerce").round(1)
            show_rb["sharpe"] = pd.to_numeric(show_rb["sharpe"], errors="coerce").round(2)
            show_rb = show_rb.rename(columns={
                "min_strength":"最低力道", "min_volume_ratio":"最低量比", "hold_days":"持有日", "stop_loss_pct":"停損%", "take_profit_pct":"停利%",
                "trades":"交易數", "total_return":"累積報酬%", "cagr":"年化報酬%", "max_drawdown":"最大回撤%", "sharpe":"Sharpe",
                "expectancy":"期望報酬%", "win_rate":"勝率%", "robustness_score":"穩健分數", "neighbor_positive_expectancy_ratio":"鄰近正期望%",
                "neighbor_positive_return_ratio":"鄰近正報酬%", "isolated_peak":"孤立高峰"
            })
            st.dataframe(show_rb, use_container_width=True, hide_index=True)
            st.caption("穩健分數是用來找參數『平台』的研究指標，不是未來報酬評分，也不是投資建議。孤立高峰代表自身結果突出，但鄰近參數沒有一起支持，需特別小心過度最佳化。")

            st.markdown("##### 力道 × 量比：鄰近年化報酬熱圖")
            hvals = sorted(pd.to_numeric(results_rb["hold_days"], errors="coerce").dropna().astype(int).unique().tolist())
            svals = sorted(pd.to_numeric(results_rb["stop_loss_pct"], errors="coerce").dropna().unique().tolist())
            tvals = sorted(pd.to_numeric(results_rb["take_profit_pct"], errors="coerce").dropna().unique().tolist())
            hcol, scol, tcol = st.columns(3)
            with hcol:
                sel_h = st.selectbox("熱圖持有日", hvals, index=0, key="rb_heat_hold")
            with scol:
                sel_s = st.selectbox("熱圖停損%", svals, index=0, key="rb_heat_stop")
            with tcol:
                sel_t = st.selectbox("熱圖停利%", tvals, index=0, key="rb_heat_take")
            pivot = heatmap_slice(results_rb, sel_h, sel_s, sel_t, metric="neighbor_median_cagr")
            if pivot.empty:
                st.info("這個切片沒有資料。")
            else:
                st.plotly_chart(robustness_heatmap_chart(pivot, "鄰近參數年化報酬中位數"), use_container_width=True)

            st.markdown("##### 單一參數敏感度")
            sensitivity = parameter_sensitivity_summary(results_rb, min_trades=min_trades_rb)
            if sensitivity:
                selected_param = st.selectbox(
                    "查看哪一個參數", list(sensitivity.keys()),
                    format_func=lambda x: {"min_strength":"最低力道", "min_volume_ratio":"最低量比", "hold_days":"持有日", "stop_loss_pct":"停損%", "take_profit_pct":"停利%"}.get(x, x),
                    key="rb_sensitivity_param",
                )
                sen_show = sensitivity[selected_param].copy()
                for c in ["累積報酬中位數", "年化報酬中位數", "期望報酬中位數", "最大回撤中位數", "正期望比例", "正報酬比例", "穩健區比例"]:
                    if c in sen_show.columns:
                        sen_show[c] = (pd.to_numeric(sen_show[c], errors="coerce") * 100).round(2)
                if "Sharpe中位數" in sen_show.columns:
                    sen_show["Sharpe中位數"] = pd.to_numeric(sen_show["Sharpe中位數"], errors="coerce").round(2)
                st.dataframe(sen_show, use_container_width=True, hide_index=True)

            st.download_button(
                "下載全部參數掃描 CSV",
                results_rb.to_csv(index=False).encode("utf-8-sig"),
                file_name="parameter_robustness_v9.csv",
                mime="text/csv",
            )

    with st.expander("v9 怎麼看，才不會把過度最佳化當成發現？"):
        st.markdown(
            "- **先看一片區域，不看單一最高點**：相鄰力道、量比、停損/停利都仍維持正期望，比某一格特別高更可信。\n"
            "- **先看樣本數**：交易太少的漂亮數字沒有太大意義。\n"
            "- **看中位數與比例**：v9 用鄰近參數的中位數與正期望比例，降低單一極端值影響。\n"
            "- **再用 v6 前向驗證**：歷史穩健只是第一關；真正是否有效，仍要看上線後沒有重調參數的 forward record。\n"
            "- **不要每週重新挑參數**：如果一直根據最新結果改規則，前向驗證也會被污染。"
        )


with model_tab:
    st.markdown("#### Walk-forward 模型檢驗")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("方向正確率", pct_text(model.accuracy))
    m2.metric("ROC AUC", "—" if model.auc is None else f"{model.auc:.3f}")
    m3.metric("Brier 分數", f"{model.brier:.3f}", help="越低越好；用來看機率預測誤差。")
    m4.metric("Walk-forward 測試樣本", f"{model.test_rows:,}")

    bt = apply_transaction_cost(model.backtest, one_way_cost=one_way_cost)
    stats = backtest_stats(bt)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("策略報酬", pct_text(stats["strategy_total"]))
    b2.metric("同期買進持有", pct_text(stats["benchmark_total"]))
    b3.metric("策略最大回撤", pct_text(stats["strategy_mdd"]))
    b4.metric("進場次數", f"{int(stats['trades'])}")

    st.plotly_chart(backtest_chart(bt), use_container_width=True)
    st.caption(
        f"規則：每 {model.retrain_every} 個交易日重新訓練一次；預測當下只使用當時已經能知道 5 日結果的舊資料。"
        f"機率 ≥ {model.threshold:.0%} 時持有下一交易日；單邊成本假設 {one_way_cost:.2%}。不含股利與滑價差異。"
    )
    st.markdown("#### 最終模型特徵重要度")
    importance = model.feature_importance.copy()
    importance["importance"] = (importance["importance"] * 100).round(1)
    st.dataframe(importance, use_container_width=True, hide_index=True)


with daily_tab:
    st.markdown("#### 每日訊號中心 / 我的自選股")
    st.caption("v7 沿用訊號工作台：近 1/3/5 日新 V、力道升溫、連續轉強、最近 V 品質，以及勝率以外的期望報酬與盈虧比。每次仍最多掃 25 檔。")

    up_col, preset_col = st.columns([2, 3])
    with up_col:
        uploaded_watchlist = st.file_uploader("匯入自選股 CSV / TXT", type=["csv", "txt"], key="watchlist_upload")
        if uploaded_watchlist is not None and st.button("匯入這份清單", key="import_watchlist"):
            try:
                imported = parse_uploaded_watchlist(uploaded_watchlist)[:25]
                if not imported:
                    st.warning("檔案裡沒有辨識到股票代號。")
                else:
                    st.session_state["watchlist_text"] = ", ".join(imported)
                    st.success(f"已匯入 {len(imported)} 檔。")
            except Exception as exc:
                st.error(f"匯入失敗：{exc}")
    with preset_col:
        st.write("快速載入內建清單")
        preset_buttons = st.columns(4)
        for idx, (label, values) in enumerate(WATCHLISTS.items()):
            if preset_buttons[idx].button(label, key=f"preset_{label}"):
                st.session_state["watchlist_text"] = ", ".join(values[:25])

    watchlist_text = st.text_area(
        "自選股代號（逗號、頓號或換行分隔）",
        key="watchlist_text",
        height=105,
        help="例如：2330, 2317, 2454。最多使用前 25 檔。",
    )
    watchlist = parse_tickers(watchlist_text)[:25]
    names_now = stock_name_map(token)
    w1, w2 = st.columns([2, 1])
    w1.write(f"目前自選股：{len(watchlist)} 檔｜" + ("、".join(watchlist) if watchlist else "尚未設定"))
    w2.download_button(
        "下載自選股 CSV",
        watchlist_csv_bytes(watchlist, names_now),
        file_name="my_tw_stock_watchlist.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("📡 掃描 v7 訊號中心", type="primary", key="scan_my_watchlist", use_container_width=True, disabled=not bool(watchlist)):
        progress = st.progress(0, text="開始掃描自選股…")
        rows = []
        errors = []
        for idx, ticker in enumerate(watchlist, start=1):
            progress.progress((idx - 1) / max(len(watchlist), 1), text=f"正在掃描 {ticker}（{idx}/{len(watchlist)}）")
            try:
                raw, _ = load_data(ticker, 2, source, token)
                rows.append(watchlist_signal_row(ticker, raw, names_now.get(ticker, KNOWN_NAMES.get(ticker, ""))))
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
        progress.progress(1.0, text="掃描完成")
        st.session_state["daily_signal_result_v6"] = pd.DataFrame(rows)
        st.session_state["daily_signal_errors_v6"] = errors

    daily_result = st.session_state.get("daily_signal_result_v6")
    if isinstance(daily_result, pd.DataFrame) and not daily_result.empty:
        latest_dates = pd.to_datetime(daily_result["資料日"], errors="coerce")
        common_latest = latest_dates.max().date().isoformat() if latest_dates.notna().any() else "—"
        n_v1 = int((daily_result["近1日新V"] == "是").sum())
        n_v3 = int((daily_result["近3日新V"] == "是").sum())
        n_v6 = int((daily_result["近5日新V"] == "是").sum())
        n_heat = int((daily_result["快速升溫"] == "是").sum())
        n_rising = int((pd.to_numeric(daily_result["連續轉強日數"], errors="coerce") >= 2).sum())
        n_a3 = int((daily_result["近3日新A"] == "是").sum())

        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("最新資料日", common_latest)
        d2.metric("近 1 日新 V", n_v1)
        d3.metric("近 3 日新 V", n_v3)
        d4.metric("近 5 日新 V", n_v6)
        d5.metric("快速升溫", n_heat)
        d6.metric("近 3 日新 A", n_a3)

        st.markdown("##### 條件篩選")
        filter_label = st.radio(
            "快速篩選",
            ["全部", "近1日新V", "近3日新V", "近5日新V", "快速升溫", "連續轉強", "近3日新A", "強多", "強空"],
            horizontal=True,
            key="daily_filter_v6",
        )
        filtered = daily_result.copy()
        if filter_label in {"近1日新V", "近3日新V", "近5日新V", "近3日新A"}:
            filtered = filtered[filtered[filter_label] == "是"]
        elif filter_label == "快速升溫":
            filtered = filtered[filtered["快速升溫"] == "是"]
        elif filter_label == "連續轉強":
            filtered = filtered[pd.to_numeric(filtered["連續轉強日數"], errors="coerce") >= 2]
        elif filter_label == "強多":
            filtered = filtered[filtered["四色狀態"] == "強多"]
        elif filter_label == "強空":
            filtered = filtered[filtered["四色狀態"] == "強空"]

        sort_mode = st.selectbox(
            "排序",
            ["V品質分數", "力道3日變化", "力道", "技術排名分數", "V後5日期望報酬%", "V盈虧比", "20日動能%"],
            key="daily_sort_v6",
        )
        filtered = filtered.sort_values(sort_mode, ascending=False, na_position="last").reset_index(drop=True)

        display_mode = st.radio("表格欄位", ["訊號中心精簡版", "完整研究欄位"], horizontal=True, key="daily_display_mode")
        show_daily = filtered.copy()
        if display_mode == "訊號中心精簡版":
            compact_cols = [
                "代號", "名稱", "資料日", "收盤", "日漲跌%", "力道", "四色狀態",
                "力道3日變化", "連續轉強日數", "快速升溫",
                "近1日新V", "近3日新V", "近5日新V", "最近V品質", "V品質分數",
                "V後5日勝率%", "V後5日期望報酬%", "V盈虧比", "V樣本", "近3日新A",
            ]
            show_daily = show_daily[[c for c in compact_cols if c in show_daily.columns]]

        numeric_cols = [
            "收盤", "日漲跌%", "力道", "力道3日變化", "力道5日變化", "連續轉強日數",
            "V品質分數", "20日動能%", "RSI", "量比", "V後5日勝率%", "A後5日勝率%",
            "V後5日期望報酬%", "A後5日期望報酬%", "V盈虧比", "A盈虧比",
            "V Profit Factor", "A Profit Factor", "技術排名分數",
        ]
        for c in numeric_cols:
            if c in show_daily.columns:
                show_daily[c] = pd.to_numeric(show_daily[c], errors="coerce").round(2)
        st.dataframe(show_daily, use_container_width=True, hide_index=True)
        st.download_button(
            "下載 v7 每日訊號總表 CSV",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="tw_stock_daily_signals_v7.csv",
            mime="text/csv",
        )

        if n_v6 > 0:
            recent_v = daily_result[daily_result["近5日新V"] == "是"].copy()
            recent_v = recent_v.sort_values(["V品質分數", "力道3日變化"], ascending=False, na_position="last")
            st.markdown("##### 近 5 日新 V：品質與歷史經濟性")
            cols = ["代號", "名稱", "最近V距今交易日", "最近V品質", "V品質分數", "力道", "力道3日變化", "V樣本", "V後5日勝率%", "V後5日期望報酬%", "V盈虧比", "V Profit Factor"]
            focus = recent_v[[c for c in cols if c in recent_v.columns]].copy()
            for c in ["V品質分數", "力道", "力道3日變化", "V後5日勝率%", "V後5日期望報酬%", "V盈虧比", "V Profit Factor"]:
                if c in focus.columns:
                    focus[c] = pd.to_numeric(focus[c], errors="coerce").round(2)
            st.dataframe(focus, use_container_width=True, hide_index=True)

        with st.expander("v7 訊號中心判定規則"):
            st.markdown(
                "- **近 1/3/5 日新 V**：最近一次 V 距最新資料日分別為 0、≤2、≤4 個交易日。\n"
                "- **連續轉強**：力道分數連續上升至少 2 個交易日。\n"
                "- **快速升溫**：力道 3 日增加至少 10 分，或 5 日增加至少 15 分。\n"
                "- **V 品質 A–D**：70 分看 V 當下技術條件，最多 30 分看該 V 以前已完成的 V 樣本；不使用該次訊號後才發生的價格。\n"
                "- **期望方向報酬**：V 以後上漲為正、A 以後下跌為正；平均包含成功與失敗訊號。\n"
                "- **盈虧比**：歷史成功訊號平均獲利 ÷ 歷史失敗訊號平均損失絕對值。"
            )

        selectable = daily_result["代號"].astype(str).tolist()
        if selectable:
            target = st.selectbox("快速切換個股分析", selectable, format_func=lambda x: f"{x} {names_now.get(x, '')}".strip(), key="daily_target")
            if st.button("載入這檔到個股分析", key="load_daily_target"):
                st.session_state["ticker"] = target
                st.rerun()

        st.caption("這些分級與排序是研究用訊號整理，不代表適合買進、賣出或保證未來報酬。不同資料來源若更新時間不同，請以各列『資料日』為準。")

    daily_errors = st.session_state.get("daily_signal_errors_v6", [])
    if daily_errors:
        with st.expander(f"有 {len(daily_errors)} 檔未成功取得資料"):
            st.write("\n".join(daily_errors))

with screener_tab:
    st.markdown("#### 免費觀察池選股排行")
    st.caption("這不是全市場掃描；為避免免費 API 額度與主機資源被一次耗盡，v7 每次最多掃 25 檔。排行是技術分數，不是投資推薦。")
    left, right = st.columns([1, 2])
    with left:
        pool = st.selectbox("內建股票池", list(WATCHLISTS.keys()), key="screen_pool")
        max_count = st.slider("最多掃描檔數", 5, 25, min(15, len(WATCHLISTS[pool])), 1)
    with right:
        custom = st.text_area("額外加入股票代號（可選）", placeholder="例如：2603, 2615, 2609", height=96)
        st.caption("可用逗號、頓號或換行分隔；重複代號會自動去除。")

    selected = merge_unique(WATCHLISTS[pool] + parse_tickers(custom))[:max_count]
    st.write("本次股票池：", "、".join(selected))

    if st.button("🔎 開始掃描", type="primary", use_container_width=True):
        progress = st.progress(0, text="開始掃描…")
        rows = []
        errors = []
        screen_names = stock_name_map(token)
        for idx, ticker in enumerate(selected, start=1):
            progress.progress((idx - 1) / max(len(selected), 1), text=f"正在掃描 {ticker}（{idx}/{len(selected)}）")
            try:
                raw, _ = load_data(ticker, 2, source, token)
                rows.append(technical_screen_row(ticker, raw, screen_names.get(ticker, KNOWN_NAMES.get(ticker, ""))))
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
        progress.progress(1.0, text="掃描完成")

        if rows:
            result = pd.DataFrame(rows).sort_values("技術排名分數", ascending=False).reset_index(drop=True)
            result.insert(0, "排名", np.arange(1, len(result) + 1))
            st.session_state["screen_result"] = result
            st.session_state["screen_errors"] = errors
        else:
            st.session_state["screen_result"] = pd.DataFrame()
            st.session_state["screen_errors"] = errors

    result = st.session_state.get("screen_result")
    if isinstance(result, pd.DataFrame) and not result.empty:
        show = result.copy()
        for c in ["收盤", "日漲跌%", "20日動能%", "力道", "RSI", "量比", "技術排名分數"]:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)
        csv = show.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載本次選股排行 CSV", csv, "tw_stock_screener_v7.csv", "text/csv")
        st.caption("技術排名分數 = 60% 力道 + 20% 20日動能 + 10% 量價 + 10% 20日區間位置；沒有使用未來資料。")

    screen_errors = st.session_state.get("screen_errors", [])
    if screen_errors:
        with st.expander(f"有 {len(screen_errors)} 檔未成功取得資料"):
            st.write("\n".join(screen_errors))

with forward_tab:
    st.markdown("#### v8 前向驗證 / Paper Tracking")
    st.caption(
        "這裡只統計 v6 起由每日快照真正記錄下來的預測；歷史回測不會混進來。"
        "快照欄位建立後不回頭改寫，未來 5/10/20 個交易日到期時才補上實際結果。"
    )

    forward_log = load_forward_log()
    current_latest_date = pd.Timestamp(latest["date"]).date()
    now_tw = pd.Timestamp.now(tz="Asia/Taipei")
    age_days = (now_tw.date() - current_latest_date).days

    health1, health2, health3, health4 = st.columns(4)
    health1.metric("目前個股資料日", current_latest_date.isoformat(), f"距今天 {max(age_days, 0)} 天")
    health2.metric("模型版本", MODEL_VERSION)

    if forward_log.empty:
        health3.metric("前向快照", "尚未開始")
        health4.metric("已完成 5 日驗證", "0")
        if age_days <= 3:
            st.success("目前個股價格資料看起來有更新；前向紀錄尚未開始累積。")
        else:
            st.warning("目前個股最新資料距今天超過 3 個日曆日，請先確認資料來源是否更新。")
        st.info(
            "第一次啟用前向追蹤時，請到 GitHub → Actions → Daily forward validation → Run workflow 手動跑一次。"
            "之後工作流程會在週一到週五台北時間 17:30 自動執行；假日若沒有新交易日，不會重複新增快照。"
        )
    else:
        latest_forward_ts = pd.to_datetime(forward_log["data_date"], errors="coerce").max()
        latest_forward_date = latest_forward_ts.date() if pd.notna(latest_forward_ts) else None
        latest_slice = forward_log[pd.to_datetime(forward_log["data_date"], errors="coerce").dt.date.eq(latest_forward_date)] if latest_forward_date else pd.DataFrame()
        done_5 = int(pd.to_numeric(forward_log.get("ret_5d"), errors="coerce").notna().sum())
        health3.metric("最新前向快照", latest_forward_date.isoformat() if latest_forward_date else "—", f"{len(latest_slice)} 檔")
        health4.metric("已完成 5 日驗證", f"{done_5:,}")

        current_snapshot_ok = bool(
            latest_forward_date
            and latest_forward_date == current_latest_date
            and (
                forward_log["stock_id"].astype(str).eq(stock_id)
                & pd.to_datetime(forward_log["data_date"], errors="coerce").dt.date.eq(current_latest_date)
                & forward_log["model_version"].astype(str).eq(MODEL_VERSION)
            ).any()
        )
        if current_snapshot_ok:
            st.success("資料健康檢查：目前個股的最新交易日已存在前向快照。")
        elif latest_forward_date and (current_latest_date - latest_forward_date).days <= 3:
            st.info("前向紀錄正在累積中；最新快照日期與目前行情資料接近。若今天剛升級，可先手動執行一次 GitHub Actions。")
        else:
            st.warning("前向快照可能落後目前行情資料，請到 GitHub Actions 檢查最近一次 Daily forward validation 是否成功。")

        versions = [str(x) for x in forward_log.get("model_version", pd.Series(dtype=str)).dropna().unique().tolist()]
        versions = sorted(versions, reverse=True)
        selected_version = st.selectbox(
            "驗證模型版本",
            versions or [MODEL_VERSION],
            index=0,
            help="模型改版後分開計分，避免新舊規則混成一個成績。",
        )
        signal_focus = st.radio("訊號方向", ["V", "A"], horizontal=True, key="forward_signal_focus")
        summary = forward_performance_summary(forward_log, signal=signal_focus, model_version=selected_version)

        st.markdown(f"##### {signal_focus} 訊號：真正的前向實績")
        if summary.empty or int(pd.to_numeric(summary.get("樣本數"), errors="coerce").fillna(0).sum()) == 0:
            st.info("還沒有到期的前向樣本。5 個交易日後會開始出現第一批正式結果。")
        else:
            show_summary = summary.copy()
            for col in ["方向勝率", "平均方向報酬", "中位方向報酬", "平均相對0050超額", "平均MFE", "平均MAE"]:
                if col in show_summary.columns:
                    show_summary[col] = (pd.to_numeric(show_summary[col], errors="coerce") * 100).round(2)
            show_summary = show_summary.rename(columns={
                "方向勝率": "方向勝率%",
                "平均方向報酬": "平均方向報酬%",
                "中位方向報酬": "中位方向報酬%",
                "平均相對0050超額": "平均相對0050超額%",
                "平均MFE": "平均MFE%",
                "平均MAE": "平均MAE%",
            })
            st.dataframe(show_summary, use_container_width=True, hide_index=True)

            five = summary[summary["期間"].eq("5日")]
            if not five.empty and int(five.iloc[0]["樣本數"]) > 0:
                row5 = five.iloc[0]
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("5日已完成樣本", f"{int(row5['樣本數'])}")
                f2.metric("5日方向勝率", pct_text(row5["方向勝率"]))
                f3.metric("5日平均方向報酬", pct_text(row5["平均方向報酬"], 2))
                f4.metric("5日相對0050超額", pct_text(row5["平均相對0050超額"], 2))

        st.markdown("##### AI 機率校準：說 60% 的時候，後來真的多常上漲？")
        calibration = forward_ai_calibration(forward_log, model_version=selected_version)
        if calibration.empty:
            st.info("需要至少有一批完成 5 個交易日的每日快照後，才會出現 AI 機率校準表。")
        else:
            cal_show = calibration.copy()
            for col in ["平均預測機率", "實際5日上漲率", "平均5日報酬"]:
                cal_show[col] = (pd.to_numeric(cal_show[col], errors="coerce") * 100).round(2)
            cal_show = cal_show.rename(columns={
                "平均預測機率": "平均預測機率%",
                "實際5日上漲率": "實際5日上漲率%",
                "平均5日報酬": "平均5日報酬%",
            })
            st.dataframe(cal_show, use_container_width=True, hide_index=True)
            st.caption("校準表使用所有每日快照，不只 V/A。樣本少時百分比會很不穩定，先看樣本數。")

        st.markdown("##### 最新每日快照")
        latest_view = latest_slice.copy()
        if not latest_view.empty:
            latest_view["AI 5日上漲機率%"] = (pd.to_numeric(latest_view["ai_up_5d_prob"], errors="coerce") * 100).round(1)
            latest_view["力道"] = pd.to_numeric(latest_view["strength_score"], errors="coerce").round(1)
            latest_view["V品質分數"] = pd.to_numeric(latest_view["v_quality_score"], errors="coerce").round(1)
            latest_view["RSI"] = pd.to_numeric(latest_view["rsi14"], errors="coerce").round(1)
            latest_view["量比"] = pd.to_numeric(latest_view["volume_ratio"], errors="coerce").round(2)
            cols = ["stock_id", "stock_name", "data_date", "close", "signal", "v_grade", "V品質分數", "力道", "strength_band", "AI 5日上漲機率%", "RSI", "量比", "source"]
            latest_view = latest_view[[c for c in cols if c in latest_view.columns]].rename(columns={
                "stock_id": "代號", "stock_name": "名稱", "data_date": "資料日", "close": "收盤",
                "signal": "當日V/A", "v_grade": "V品質", "strength_band": "四色狀態", "source": "資料來源",
            })
            st.dataframe(latest_view, use_container_width=True, hide_index=True)

        st.markdown("##### 已記錄的 V/A 訊號日誌")
        siglog = forward_log[forward_log["signal"].astype(str).isin(["V", "A"])].copy()
        siglog = siglog[siglog["model_version"].astype(str).eq(selected_version)]
        siglog = siglog.sort_values("data_date", ascending=False).head(100)
        if siglog.empty:
            st.info("目前前向期間尚未出現 V/A。每日快照仍會照常保存，供 AI 機率校準使用。")
        else:
            view = siglog.copy()
            percent_cols = ["ai_up_5d_prob", "ret_5d", "ret_10d", "ret_20d", "excess_5d", "mfe_5d", "mae_5d"]
            for col in percent_cols:
                if col in view.columns:
                    view[col] = (pd.to_numeric(view[col], errors="coerce") * 100).round(2)
            columns = [
                "data_date", "stock_id", "stock_name", "signal", "v_grade", "v_quality_score", "strength_score",
                "ai_up_5d_prob", "ret_5d", "ret_10d", "ret_20d", "excess_5d", "mfe_5d", "mae_5d", "model_version",
            ]
            view = view[[c for c in columns if c in view.columns]].rename(columns={
                "data_date": "訊號日", "stock_id": "代號", "stock_name": "名稱", "signal": "V/A",
                "v_grade": "V品質", "v_quality_score": "V品質分數", "strength_score": "力道",
                "ai_up_5d_prob": "AI機率%", "ret_5d": "5日後%", "ret_10d": "10日後%", "ret_20d": "20日後%",
                "excess_5d": "5日相對0050超額%", "mfe_5d": "5日MFE%", "mae_5d": "5日MAE%", "model_version": "模型版本",
            })
            st.dataframe(view, use_container_width=True, hide_index=True)

        st.download_button(
            "下載完整前向驗證紀錄 CSV",
            forward_log.to_csv(index=False).encode("utf-8-sig"),
            file_name="forward_signals_v7.csv",
            mime="text/csv",
        )

    with st.expander("前向驗證為什麼這樣做？"):
        st.markdown(
            "- **快照與答案分開**：今天只保存今天已知的價格、訊號、力道與 AI 機率。\n"
            "- **未來到期才補答案**：5/10/20 個交易日後才填入實際報酬，不用未來資料改寫今天的預測。\n"
            "- **模型版本鎖定**：目前模型仍為 `v6-model-1.0`（v7 未修改預測演算法）；之後修改演算法會換版本，舊成績不混入。\n"
            "- **0050 基準**：同期間與 0050 比較，避免只因整體大盤上漲就誤以為訊號特別有效。\n"
            "- **MFE / MAE**：MFE 是訊號後曾出現的最大有利漲幅，MAE 是最大不利跌幅，兩者皆以訊號日收盤為基準。"
        )


with notification_tab:
    st.markdown("#### 🔔 v9 通知中心")
    st.caption(
        "GitHub Actions 每日收盤後偵測新 V/A、A 級 V、力道快速升溫，以及你自行登錄研究部位的停損／停利。"
        "就算沒有設定 Telegram 或 Discord，事件仍會寫進 data/notification_events.csv。"
    )

    notification_log = load_notification_log()
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("通知事件總數", f"{len(notification_log):,}" if not notification_log.empty else "0")
    if notification_log.empty:
        n2.metric("最新事件日", "—")
        n3.metric("Telegram", "未判定")
        n4.metric("Discord", "未判定")
        st.info("目前還沒有通知事件。第一次 v7 Actions 執行後，有符合條件的事件才會開始累積。")
    else:
        latest_event_date = pd.to_datetime(notification_log["data_date"], errors="coerce").max()
        n2.metric("最新事件日", latest_event_date.date().isoformat() if pd.notna(latest_event_date) else "—")
        tg = notification_log.get("telegram_status", pd.Series(dtype=str)).astype(str)
        dc = notification_log.get("discord_status", pd.Series(dtype=str)).astype(str)
        n3.metric("Telegram 最近狀態", tg.iloc[-1] if len(tg) else "—")
        n4.metric("Discord 最近狀態", dc.iloc[-1] if len(dc) else "—")

        event_names = {
            "A_GRADE_V": "A 級新 V",
            "NEW_V": "新 V",
            "NEW_A": "新 A",
            "RAPID_WARMING": "力道快速升溫",
            "FORWARD_5D_COMPLETE": "5 日前向結果完成",
            "POSITION_STOP": "研究部位停損",
            "POSITION_TAKE": "研究部位停利",
        }
        event_types = [str(x) for x in notification_log.get("event_type", pd.Series(dtype=str)).dropna().unique().tolist()]
        selected_types = st.multiselect(
            "篩選事件類型",
            event_types,
            default=event_types,
            format_func=lambda x: event_names.get(x, x),
        )
        view = notification_log.copy()
        if selected_types:
            view = view[view["event_type"].astype(str).isin(selected_types)]
        view = view.sort_values(["data_date", "created_at"], ascending=False).head(200)
        show_cols = ["data_date", "stock_id", "stock_name", "event_type", "title", "message", "telegram_status", "discord_status"]
        view = view[[c for c in show_cols if c in view.columns]].rename(columns={
            "data_date": "事件日", "stock_id": "代號", "stock_name": "名稱", "event_type": "事件類型",
            "title": "標題", "message": "內容", "telegram_status": "Telegram", "discord_status": "Discord",
        })
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "下載通知事件 CSV",
            notification_log.to_csv(index=False).encode("utf-8-sig"),
            file_name="notification_events_v7.csv",
            mime="text/csv",
        )

    st.markdown("##### 目前通知規則")
    settings_table = load_small_csv("notification_settings.csv")
    if settings_table.empty:
        st.warning("找不到 notification_settings.csv。GitHub Actions 仍可執行，但會使用程式預設值。")
    else:
        rules = settings_table.iloc[0]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("新 V / A", "啟用" if str(rules.get("notify_new_v", "1")) == "1" and str(rules.get("notify_new_a", "1")) == "1" else "部分/停用")
        r2.metric("A 級 V", "啟用" if str(rules.get("notify_a_grade_v", "1")) == "1" else "停用")
        r3.metric("快速升溫", f"3日 +{rules.get('rapid_3d_threshold', '10')} / 5日 +{rules.get('rapid_5d_threshold', '15')}")
        r4.metric("停損/停利", "啟用" if str(rules.get("notify_stop_take", "1")) == "1" else "停用")
        with st.expander("查看 notification_settings.csv 原始設定"):
            st.dataframe(settings_table, use_container_width=True, hide_index=True)

    st.markdown("##### 研究部位停損／停利")
    positions_table = load_small_csv("paper_positions.csv")
    if positions_table.empty:
        st.info(
            "目前沒有研究部位。若要啟用，可在 GitHub 編輯 paper_positions.csv。"
            "這是每日收盤後用日 K 高低價檢查的研究提醒，不是盤中即時交易警報。"
        )
    else:
        st.dataframe(positions_table, use_container_width=True, hide_index=True)
        st.caption("同一天同時碰到停損與停利時，v7 採較保守的『停損先發生』假設。")

    with st.expander("如何開啟免費 Telegram / Discord 推播"):
        st.markdown(
            "**Telegram**：在 GitHub Repository → Settings → Secrets and variables → Actions，加入 "
            "`TELEGRAM_BOT_TOKEN` 與 `TELEGRAM_CHAT_ID`。\n\n"
            "**Discord**：加入 `DISCORD_WEBHOOK_URL`。\n\n"
            "完成後到 GitHub → Actions → **Test notifications** → Run workflow。"
            "至少一個管道設定正確時，會收到 v7 測試訊息。Secrets 不會寫進 CSV 或顯示在這個網站。"
        )


with data_tab:
    show = df[[
        "date", "open", "high", "low", "close", "volume", "ma20", "ma60", "rsi14",
        "macd", "macd_signal", "strength_score", "strength_band", "signal"
    ]].sort_values("date", ascending=False)
    st.dataframe(show, use_container_width=True, hide_index=True)
    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載目前分析資料 CSV", data=csv, file_name=f"{stock_id}_analysis_v9.csv", mime="text/csv")

st.divider()
st.markdown(
    "**重要聲明：** 本工具為教育與研究用途。模型與技術訊號可能失效；歷史回測不代表未來績效。"
    "本版本未納入完整交易稅費結構、滑價、股利、財報、新聞與籌碼資訊。"
)
