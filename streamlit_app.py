from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_engine import (
    add_indicators,
    fetch_stock_data,
    max_drawdown,
    normalize_stock_id,
    recent_support_resistance,
    strength_label,
    train_prediction_model,
)


st.set_page_config(
    page_title="免費台股 AI 多空分析 v1",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(stock_id: str, years: int, source: str, token: str):
    return fetch_stock_data(stock_id, years=years, source=source, finmind_token=token)


@st.cache_data(ttl=3600, show_spinner=False)
def prepare_data(df: pd.DataFrame):
    enriched = add_indicators(df)
    model_result = train_prediction_model(enriched)
    return enriched, model_result


def pct_text(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


def price_chart(df: pd.DataFrame, days: int = 180) -> go.Figure:
    p = df.tail(days)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=p["date"],
            open=p["open"],
            high=p["high"],
            low=p["low"],
            close=p["close"],
            name="K線",
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
        )
    )
    fig.add_trace(go.Scatter(x=p["date"], y=p["ma20"], name="MA20", mode="lines"))
    fig.add_trace(go.Scatter(x=p["date"], y=p["ma60"], name="MA60", mode="lines"))
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        hovermode="x unified",
    )
    return fig


def strength_candle_chart(df: pd.DataFrame, days: int = 180) -> go.Figure:
    p = df.tail(days).copy()
    fig = go.Figure()
    bands = [
        (75, 101, "強多", "#b2182b"),
        (60, 75, "偏多", "#ef8a62"),
        (40, 60, "盤整", "#bdbdbd"),
        (0, 40, "偏空", "#1b7837"),
    ]
    for lower, upper, label, color in bands:
        mask = (p["strength_score"] >= lower) & (p["strength_score"] < upper)
        q = p.loc[mask]
        if q.empty:
            continue
        fig.add_trace(
            go.Candlestick(
                x=q["date"],
                open=q["open"],
                high=q["high"],
                low=q["low"],
                close=q["close"],
                name=label,
                increasing_line_color=color,
                increasing_fillcolor=color,
                decreasing_line_color=color,
                decreasing_fillcolor=color,
            )
        )

    buy = p[p["signal"] == "V"]
    sell = p[p["signal"] == "A"]
    if not buy.empty:
        fig.add_trace(
            go.Scatter(
                x=buy["date"], y=buy["low"] * 0.985, mode="markers+text",
                text=["V"] * len(buy), textposition="bottom center",
                marker=dict(symbol="triangle-up", size=12), name="V 多訊號"
            )
        )
    if not sell.empty:
        fig.add_trace(
            go.Scatter(
                x=sell["date"], y=sell["high"] * 1.015, mode="markers+text",
                text=["A"] * len(sell), textposition="top center",
                marker=dict(symbol="triangle-down", size=12), name="A 空訊號"
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        hovermode="x unified",
    )
    return fig


def indicator_chart(df: pd.DataFrame, days: int = 180) -> go.Figure:
    p = df.tail(days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["date"], y=p["rsi14"], name="RSI(14)", mode="lines"))
    fig.add_hline(y=70, line_dash="dash", annotation_text="70")
    fig.add_hline(y=50, line_dash="dot", annotation_text="50")
    fig.add_hline(y=30, line_dash="dash", annotation_text="30")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[0, 100])
    return fig


def backtest_chart(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["strategy_curve"], name="模型策略", mode="lines"))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["benchmark_curve"], name="買進持有", mode="lines"))
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis_title="資產倍數",
        legend_orientation="h",
        hovermode="x unified",
    )
    return fig


st.title("📈 免費台股 AI 多空分析系統 v1")
st.caption("日 K 版本｜免費資料來源｜技術指標 + 機器學習機率 + 留出樣本回測")

with st.sidebar:
    st.header("分析設定")
    with st.form("query_form"):
        ticker_input = st.text_input("股票代號", value=st.session_state.get("ticker", "2330"), help="例如：2330、2317、2454")
        years = st.select_slider("歷史資料期間", options=[2, 3, 5, 8], value=st.session_state.get("years", 3))
        source_options = ["自動（FinMind → Yahoo 備援）", "只用 FinMind", "只用 Yahoo Finance"]
        saved_source = st.session_state.get("source_label", source_options[0])
        saved_index = source_options.index(saved_source) if saved_source in source_options else 0
        source_label = st.selectbox("資料來源", source_options, index=saved_index)
        token = st.text_input(
            "FinMind Token（選填）",
            value=st.session_state.get("finmind_token", ""),
            type="password",
            help="不填也可使用匿名免費額度；免費註冊 token 可提高額度。",
        )
        submitted = st.form_submit_button("開始分析", use_container_width=True)

    if submitted or "ticker" not in st.session_state:
        st.session_state["ticker"] = ticker_input.strip()
        st.session_state["years"] = years
        st.session_state["source_label"] = source_label
        st.session_state["finmind_token"] = token

    st.divider()
    st.caption("v1 是研究工具，不是投資建議。模型只使用歷史價格/成交量特徵，沒有新聞、財報或籌碼資料。")

stock_input = st.session_state.get("ticker", "2330")
years = int(st.session_state.get("years", 3))
source_label = st.session_state.get("source_label", "自動（FinMind → Yahoo 備援）")
token = st.session_state.get("finmind_token", "")
source_map = {
    "自動（FinMind → Yahoo 備援）": "auto",
    "只用 FinMind": "finmind",
    "只用 Yahoo Finance": "yahoo",
}
source = source_map.get(source_label, "auto")
stock_id = normalize_stock_id(stock_input)

try:
    with st.spinner(f"正在取得 {stock_id} 的歷史資料並建立模型…"):
        raw_df, used_source = load_data(stock_input, years, source, token)
        df, model = prepare_data(raw_df)
except Exception as exc:
    st.error(f"目前無法完成分析：{exc}")
    st.info("可以先確認股票代號；若 FinMind 達到匿名流量上限，可改用 Yahoo Finance，或免費申請 FinMind Token。")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2]
support, resistance = recent_support_resistance(df, 20)
score = float(latest["strength_score"])
prob = model.probability_up_5d
latest_signal = latest["signal"] or "—"
trend = strength_label(score)
price_delta = float(latest["close"] / prev["close"] - 1)

st.subheader(f"{stock_id}｜最新分析：{latest['date'].date()}")
st.caption(f"資料來源：{used_source}｜共 {len(df):,} 個交易日")

c1, c2, c3, c4 = st.columns(4)
c1.metric("收盤價", f"{latest['close']:.2f}", pct_text(price_delta))
c2.metric("多空力道", f"{score:.0f}/100", trend)
c3.metric("未來 5 日上漲機率", f"{prob * 100:.1f}%", "機器學習估計")
c4.metric("最新 V/A 訊號", latest_signal, "V=轉強、A=轉弱")

s1, s2, s3, s4 = st.columns(4)
s1.metric("20 日支撐參考", f"{support:.2f}")
s2.metric("20 日壓力參考", f"{resistance:.2f}")
s3.metric("RSI(14)", f"{latest['rsi14']:.1f}")
s4.metric("成交量 / 20日均量", f"{latest['volume_ratio']:.2f}x")

if prob >= 0.60 and score >= 60:
    st.success("目前屬於「技術面偏多 + 模型機率偏多」的同向狀態。這不是買進建議，請搭配風險與部位管理。")
elif prob <= 0.40 and score <= 40:
    st.warning("目前屬於「技術面偏空 + 模型機率偏低」的同向狀態。這不是賣出建議。")
else:
    st.info("技術力道與模型機率目前沒有形成明顯同向訊號，可視為中性/分歧狀態。")

price_tab, strength_tab, indicator_tab, model_tab, data_tab = st.tabs(
    ["K 線", "力道 K + V/A", "技術指標", "AI 與回測", "原始資料"]
)

with price_tab:
    st.plotly_chart(price_chart(df), use_container_width=True)
    st.caption("台股慣例：紅 K 代表上漲、綠 K 代表下跌。MA20 / MA60 為 20 / 60 日均線。")

with strength_tab:
    st.plotly_chart(strength_candle_chart(df), use_container_width=True)
    st.caption("力道 K 是本專案自行設計的規則分數，不是參考網站的專有公式。V 表示力道由弱轉強穿越 60；A 表示由強轉弱跌破 40。")

with indicator_tab:
    st.plotly_chart(indicator_chart(df), use_container_width=True)
    table = df[["date", "close", "ma20", "ma60", "rsi14", "macd_hist", "volume_ratio", "strength_score"]].tail(20).copy()
    table = table.sort_values("date", ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)

with model_tab:
    st.markdown("#### 模型測試")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("測試正確率", pct_text(model.accuracy))
    m2.metric("ROC AUC", "—" if model.auc is None else f"{model.auc:.3f}")
    m3.metric("訓練樣本", f"{model.train_rows:,}")
    m4.metric("測試樣本", f"{model.test_rows:,}")

    bt = model.backtest
    strategy_total = float(bt["strategy_curve"].iloc[-1] - 1)
    benchmark_total = float(bt["benchmark_curve"].iloc[-1] - 1)
    strategy_mdd = max_drawdown(bt["strategy_curve"])
    active_days = int(bt["position"].sum())

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("測試期策略報酬", pct_text(strategy_total))
    b2.metric("同期買進持有", pct_text(benchmark_total))
    b3.metric("策略最大回撤", pct_text(strategy_mdd))
    b4.metric("持倉交易日", f"{active_days}")

    st.plotly_chart(backtest_chart(bt), use_container_width=True)
    st.caption(
        f"回測規則：模型預估未來 5 日上漲機率 ≥ {model.threshold:.0%} 時，下一交易日持有；否則持有現金。"
        "測試資料位於時間序列最後約 25%，模型只以更早期資料訓練。未計交易成本、滑價、稅費與股利。"
    )

    st.markdown("#### 模型最常使用的特徵")
    importance = model.feature_importance.copy()
    importance["importance"] = (importance["importance"] * 100).round(1)
    st.dataframe(importance, use_container_width=True, hide_index=True)

with data_tab:
    show = df[[
        "date", "open", "high", "low", "close", "volume", "ma20", "ma60",
        "rsi14", "macd", "macd_signal", "strength_score", "signal"
    ]].sort_values("date", ascending=False)
    st.dataframe(show, use_container_width=True, hide_index=True)
    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載目前分析資料 CSV",
        data=csv,
        file_name=f"{stock_id}_analysis_v1.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    "**重要聲明：** 本工具為教育與研究用途。機器學習模型可能失效，歷史回測不代表未來績效；"
    "本版本未納入交易成本、除權息調整、即時報價、新聞、基本面與籌碼資料。"
)
