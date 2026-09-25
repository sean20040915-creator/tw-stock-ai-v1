from __future__ import annotations

import math
from io import BytesIO
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
    page_title="免費台股 AI 多空分析 v5",
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

st.title("📈 免費台股 AI 多空分析系統 v5")
st.caption("策略研究實驗室｜每日訊號中心｜V 品質｜停損停利回測｜期望報酬｜Walk-forward")

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
    st.caption("v5 為研究工具，不是投資建議。『AI』是歷史價格/成交量的機器學習機率，不是保證預測。")

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

summary_tab, strength_tab, signal_tab, strategy_tab, model_tab, daily_tab, screener_tab, data_tab = st.tabs(
    ["個股總覽", "四色力道 K + V/A", "V/A 歷史統計", "策略研究實驗室", "AI / Walk-forward", "每日訊號中心", "觀察池選股排行", "資料"]
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
    st.caption("v5 沿用四色規則：強多(≥70)、偏多(50–69)、偏空(30–49)、強空(<30)。V/A 會要求 MA20 與 MACD 同向確認。")
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
            file_name=f"{stock_id}_va_history_v5.csv",
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
            st.session_state["strategy_lab_v5"] = {
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

    lab = st.session_state.get("strategy_lab_v5")
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
                file_name=f"{stock_id}_strategy_lab_v5.csv",
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

    with st.expander("v5 策略回測規則與限制"):
        st.markdown(
            "- **進場**：V 訊號當天收盤後才知道，因此下一交易日開盤進場，避免偷看未來。\n"
            "- **篩選**：品質、力道、量比、20 日動能都只使用 V 訊號當天已知資料。\n"
            "- **部位**：一次只持有一筆、假設每筆使用全部研究資金；持倉中出現的新 V 會略過。\n"
            "- **停損 / 停利**：用日 K 高低價判斷；若同一天兩者都被觸及，採較保守的『停損先發生』假設。\n"
            "- **跳空**：若開盤已越過停損/停利價，使用實際開盤價出場。\n"
            "- **成本**：進場與出場各扣一次側欄設定的單邊成本。尚未模擬滑價、股利與完整稅費差異。"
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
    st.caption("v5 把自選股掃描升級成訊號工作台：近 1/3/5 日新 V、力道升溫、連續轉強、最近 V 品質，以及勝率以外的期望報酬與盈虧比。每次仍最多掃 25 檔。")

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

    if st.button("📡 掃描 v5 訊號中心", type="primary", key="scan_my_watchlist", use_container_width=True, disabled=not bool(watchlist)):
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
        st.session_state["daily_signal_result_v5"] = pd.DataFrame(rows)
        st.session_state["daily_signal_errors_v5"] = errors

    daily_result = st.session_state.get("daily_signal_result_v5")
    if isinstance(daily_result, pd.DataFrame) and not daily_result.empty:
        latest_dates = pd.to_datetime(daily_result["資料日"], errors="coerce")
        common_latest = latest_dates.max().date().isoformat() if latest_dates.notna().any() else "—"
        n_v1 = int((daily_result["近1日新V"] == "是").sum())
        n_v3 = int((daily_result["近3日新V"] == "是").sum())
        n_v5 = int((daily_result["近5日新V"] == "是").sum())
        n_heat = int((daily_result["快速升溫"] == "是").sum())
        n_rising = int((pd.to_numeric(daily_result["連續轉強日數"], errors="coerce") >= 2).sum())
        n_a3 = int((daily_result["近3日新A"] == "是").sum())

        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("最新資料日", common_latest)
        d2.metric("近 1 日新 V", n_v1)
        d3.metric("近 3 日新 V", n_v3)
        d4.metric("近 5 日新 V", n_v5)
        d5.metric("快速升溫", n_heat)
        d6.metric("近 3 日新 A", n_a3)

        st.markdown("##### 條件篩選")
        filter_label = st.radio(
            "快速篩選",
            ["全部", "近1日新V", "近3日新V", "近5日新V", "快速升溫", "連續轉強", "近3日新A", "強多", "強空"],
            horizontal=True,
            key="daily_filter_v5",
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
            key="daily_sort_v5",
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
            "下載 v5 每日訊號總表 CSV",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="tw_stock_daily_signals_v5.csv",
            mime="text/csv",
        )

        if n_v5 > 0:
            recent_v = daily_result[daily_result["近5日新V"] == "是"].copy()
            recent_v = recent_v.sort_values(["V品質分數", "力道3日變化"], ascending=False, na_position="last")
            st.markdown("##### 近 5 日新 V：品質與歷史經濟性")
            cols = ["代號", "名稱", "最近V距今交易日", "最近V品質", "V品質分數", "力道", "力道3日變化", "V樣本", "V後5日勝率%", "V後5日期望報酬%", "V盈虧比", "V Profit Factor"]
            focus = recent_v[[c for c in cols if c in recent_v.columns]].copy()
            for c in ["V品質分數", "力道", "力道3日變化", "V後5日勝率%", "V後5日期望報酬%", "V盈虧比", "V Profit Factor"]:
                if c in focus.columns:
                    focus[c] = pd.to_numeric(focus[c], errors="coerce").round(2)
            st.dataframe(focus, use_container_width=True, hide_index=True)

        with st.expander("v5 訊號中心判定規則"):
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

    daily_errors = st.session_state.get("daily_signal_errors_v5", [])
    if daily_errors:
        with st.expander(f"有 {len(daily_errors)} 檔未成功取得資料"):
            st.write("\n".join(daily_errors))

with screener_tab:
    st.markdown("#### 免費觀察池選股排行")
    st.caption("這不是全市場掃描；為避免免費 API 額度與主機資源被一次耗盡，v5 每次最多掃 25 檔。排行是技術分數，不是投資推薦。")
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
        st.download_button("下載本次選股排行 CSV", csv, "tw_stock_screener_v5.csv", "text/csv")
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
    st.download_button("下載目前分析資料 CSV", data=csv, file_name=f"{stock_id}_analysis_v5.csv", mime="text/csv")

st.divider()
st.markdown(
    "**重要聲明：** 本工具為教育與研究用途。模型與技術訊號可能失效；歷史回測不代表未來績效。"
    "本版本未納入完整交易稅費結構、滑價、股利、財報、新聞與籌碼資訊。"
)
