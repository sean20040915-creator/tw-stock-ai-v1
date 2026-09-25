from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stock_engine import (
    add_indicators,
    apply_transaction_cost,
    backtest_stats,
    fetch_stock_data,
    fetch_stock_info,
    normalize_stock_id,
    recent_support_resistance,
    strength_label,
    technical_screen_row,
    train_prediction_model,
)


st.set_page_config(
    page_title="免費台股 AI 多空分析 v2",
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
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


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


st.title("📈 免費台股 AI 多空分析系統 v2")
st.caption("四色力道 K｜V/A 轉折｜Walk-forward｜免費觀察池選股")

with st.sidebar:
    st.header("個股分析設定")
    with st.form("query_form"):
        ticker_input = st.text_input("股票代號", value=st.session_state.get("ticker", "2330"), help="例如：2330、2317、2454")
        years = st.select_slider("歷史資料期間", options=[3, 5, 8], value=st.session_state.get("years", 3))
        source_options = ["自動（FinMind → Yahoo 備援）", "只用 FinMind", "只用 Yahoo Finance"]
        saved_source = st.session_state.get("source_label", source_options[0])
        saved_index = source_options.index(saved_source) if saved_source in source_options else 0
        source_label = st.selectbox("資料來源", source_options, index=saved_index)
        token = st.text_input("FinMind Token（選填）", value=st.session_state.get("finmind_token", ""), type="password",
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
    st.caption("v2 為研究工具，不是投資建議。『AI』是歷史價格/成交量的機器學習機率，不是保證預測。")

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

summary_tab, strength_tab, model_tab, screener_tab, data_tab = st.tabs(
    ["個股總覽", "四色力道 K + V/A", "AI / Walk-forward", "觀察池選股排行", "資料"]
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
    st.caption("v2 四色：強多(≥70)、偏多(50–69)、偏空(30–49)、強空(<30)。V/A 還會要求 MA20 與 MACD 同向確認，因此比 v1 更少、更嚴格。")
    st.caption("四色力道、翻轉線與 V/A 規則都是本專案自行設計，不是參考網站的專有公式。")

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

with screener_tab:
    st.markdown("#### 免費觀察池選股排行")
    st.caption("這不是全市場掃描；為避免免費 API 額度與主機資源被一次耗盡，v2 每次最多掃 25 檔。排行是技術分數，不是投資推薦。")
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
        st.download_button("下載本次選股排行 CSV", csv, "tw_stock_screener_v2.csv", "text/csv")
        st.caption("技術排名分數 = 60% 力道 + 20% 20日動能 + 10% 量價 + 10% 20日區間位置；沒有使用未來資料。")

    screen_errors = st.session_state.get("screen_errors", [])
    if screen_errors:
        with st.expander(f"有 {len(screen_errors)} 檔未成功取得資料"):
            st.write("\n".join(screen_errors))

with data_tab:
    show = df[[
        "date", "open", "high", "low", "close", "volume", "ma20", "ma60", "rsi14",
        "macd", "macd_signal", "strength_score", "strength_band", "signal"
    ]].sort_values("date", ascending=False)
    st.dataframe(show, use_container_width=True, hide_index=True)
    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載目前分析資料 CSV", data=csv, file_name=f"{stock_id}_analysis_v2.csv", mime="text/csv")

st.divider()
st.markdown(
    "**重要聲明：** 本工具為教育與研究用途。模型與技術訊號可能失效；歷史回測不代表未來績效。"
    "本版本未納入完整交易稅費結構、滑價、股利、財報、新聞與籌碼資訊。"
)
