from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FEATURES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "ma20_ratio",
    "ma60_ratio",
    "rsi14",
    "macd_hist",
    "atr_pct",
    "volatility20",
    "volume_ratio",
    "range_pct",
    "breakout_pos20",
    "strength_score",
]


@dataclass
class ModelResult:
    probability_up_5d: float
    accuracy: float
    auc: Optional[float]
    brier: float
    train_rows: int
    test_rows: int
    threshold: float
    retrain_every: int
    backtest: pd.DataFrame
    feature_importance: pd.DataFrame


def normalize_stock_id(raw: str) -> str:
    text = (raw or "2330").strip().upper()
    for suffix in (".TW", ".TWO"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"資料缺少欄位：{', '.join(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["volume"] = out["volume"].fillna(0)
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)
    return out


def fetch_finmind(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": normalize_stock_id(stock_id),
        "start_date": start_date,
        "end_date": end_date,
    }
    headers = {}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"

    response = requests.get(FINMIND_URL, params=params, headers=headers, timeout=25)
    if response.status_code == 402:
        raise ValueError("FinMind 免費 API 額度已達上限")
    if response.status_code == 403:
        raise ValueError("FinMind 暫時拒絕此連線，請稍後再試或改用 Yahoo Finance")
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not data:
        msg = payload.get("msg") or "FinMind 沒有回傳資料"
        raise ValueError(msg)

    df = pd.DataFrame(data).rename(
        columns={
            "max": "high",
            "min": "low",
            "Trading_Volume": "volume",
        }
    )
    return _clean_ohlcv(df[["date", "open", "high", "low", "close", "volume"]])


def fetch_yahoo(stock_id: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    import yfinance as yf

    raw = (stock_id or "2330").strip().upper()
    candidates = [raw] if raw.endswith((".TW", ".TWO")) else [f"{raw}.TW", f"{raw}.TWO"]
    end_plus_one = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_error = None

    for symbol in candidates:
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_plus_one,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                multi_level_index=False,
                timeout=15,
            )
            if data is None or data.empty:
                continue

            data = data.reset_index().rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            cleaned = _clean_ohlcv(data[["date", "open", "high", "low", "close", "volume"]])
            if len(cleaned) >= 60:
                return cleaned, symbol
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc

    if last_error:
        raise ValueError(f"Yahoo Finance 取資料失敗：{last_error}")
    raise ValueError("Yahoo Finance 找不到這個台股代號。請確認代號，例如 2330。")


def fetch_stock_data(
    stock_id: str,
    years: int = 3,
    source: str = "auto",
    finmind_token: str = "",
) -> tuple[pd.DataFrame, str]:
    end = date.today()
    start = end - timedelta(days=int(years * 365.25) + 150)
    start_s = start.isoformat()
    end_s = end.isoformat()

    errors: list[str] = []
    source = source.lower()

    if source in ("auto", "finmind"):
        try:
            return fetch_finmind(stock_id, start_s, end_s, finmind_token), "FinMind"
        except Exception as exc:
            errors.append(f"FinMind：{exc}")
            if source == "finmind":
                raise

    if source in ("auto", "yahoo"):
        try:
            df, symbol = fetch_yahoo(stock_id, start_s, end_s)
            return df, f"Yahoo Finance ({symbol})"
        except Exception as exc:
            errors.append(f"Yahoo：{exc}")
            if source == "yahoo":
                raise

    raise ValueError("；".join(errors) or "無法取得股票資料")


def fetch_stock_info(token: str = "") -> pd.DataFrame:
    """一次取得台股名稱/市場/產業；失敗時交由 UI 使用內建名稱。"""
    headers = {}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    response = requests.get(
        FINMIND_URL,
        params={"dataset": "TaiwanStockInfo"},
        headers=headers,
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise ValueError(payload.get("msg") or "FinMind 沒有回傳股票清單")
    out = pd.DataFrame(data)
    cols = [c for c in ["stock_id", "stock_name", "industry_category", "type", "date"] if c in out.columns]
    out = out[cols].copy()
    out["stock_id"] = out["stock_id"].astype(str)
    if "date" in out.columns:
        out = out.sort_values("date").drop_duplicates("stock_id", keep="last")
    else:
        out = out.drop_duplicates("stock_id", keep="last")
    return out.reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)
    out["ma5"] = close.rolling(5).mean()
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ma20_ratio"] = close / out["ma20"] - 1
    out["ma60_ratio"] = close / out["ma60"] - 1

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))
    out.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100
    out.loc[(avg_gain == 0) & (avg_loss > 0), "rsi14"] = 0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = true_range.rolling(14).mean()
    out["atr_pct"] = out["atr14"] / close
    out["volatility20"] = out["ret_1"].rolling(20).std()
    out["volume_ma20"] = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / out["volume_ma20"].replace(0, np.nan)
    out["range_pct"] = (out["high"] - out["low"]) / close.replace(0, np.nan)

    low20 = out["low"].rolling(20).min()
    high20 = out["high"].rolling(20).max()
    out["breakout_pos20"] = (close - low20) / (high20 - low20).replace(0, np.nan)

    # v2 多空力道：只使用當日與過去資料，不偷看未來。
    score = pd.Series(50.0, index=out.index)
    score += np.where(close > out["ma20"], 10, -10)
    score += np.where(out["ma20"] > out["ma60"], 10, -10)
    score += np.where(out["ma5"] > out["ma20"], 6, -6)
    score += np.where(out["macd_hist"] > 0, 8, -8)
    score += np.where(out["rsi14"] >= 55, 7, np.where(out["rsi14"] <= 45, -7, 0))
    score += np.where(out["ret_5"] > 0, 5, -5)
    score += np.where(out["ret_20"] > 0, 5, -5)
    score += np.where(
        (out["volume_ratio"] > 1.15) & (out["ret_1"] > 0),
        5,
        np.where((out["volume_ratio"] > 1.15) & (out["ret_1"] < 0), -5, 0),
    )
    score += np.where(out["breakout_pos20"] >= 0.8, 4, np.where(out["breakout_pos20"] <= 0.2, -4, 0))
    out["strength_score"] = score.clip(0, 100)

    # 四色狀態：強多/偏多/偏空/強空，不另外設灰色區。
    out["strength_band"] = pd.cut(
        out["strength_score"],
        bins=[-0.01, 30, 50, 70, 100.01],
        labels=["強空", "偏空", "偏多", "強多"],
        include_lowest=True,
    ).astype("object")

    # V/A：除了分數穿越，也要求價格與 MACD 同向確認，降低過度頻繁訊號。
    prev_score = out["strength_score"].shift(1)
    out["signal"] = ""
    v_cond = (
        (out["strength_score"] >= 55)
        & (prev_score < 55)
        & (close > out["ma20"])
        & (out["macd_hist"] > 0)
    )
    a_cond = (
        (out["strength_score"] <= 45)
        & (prev_score > 45)
        & (close < out["ma20"])
        & (out["macd_hist"] < 0)
    )
    out.loc[v_cond, "signal"] = "V"
    out.loc[a_cond, "signal"] = "A"

    # 趨勢翻轉狀態，供圖表顯示 EMA20 紅/綠線段。
    out["trend_state"] = np.where(out["strength_score"] >= 50, 1, -1)
    return out


def strength_label(score: float) -> str:
    if score >= 70:
        return "強多"
    if score >= 50:
        return "偏多"
    if score >= 30:
        return "偏空"
    return "強空"


def recent_support_resistance(df: pd.DataFrame, window: int = 20) -> tuple[float, float]:
    tail = df.tail(window)
    return float(tail["low"].min()), float(tail["high"].max())


def _new_model(n_estimators: int = 120) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=7,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def train_prediction_model(
    df: pd.DataFrame,
    threshold: float = 0.55,
    retrain_every: int = 30,
    min_train: int = 260,
) -> ModelResult:
    """Expanding-window walk-forward。每次只用當時已經能知道結果的樣本。"""
    work = df.copy().reset_index(drop=True)
    work["target_ret_5"] = work["close"].shift(-5) / work["close"] - 1
    work["target"] = np.where(
        work["target_ret_5"].notna(),
        (work["target_ret_5"] > 0).astype(int),
        np.nan,
    )
    work["next_day_ret"] = work["close"].pct_change().shift(-1)

    model_df = work.dropna(subset=FEATURES + ["target"]).copy().reset_index(drop=True)
    if len(model_df) < min_train + 90:
        raise ValueError("歷史資料不足以做 walk-forward；請把資料期間調成 3 年以上。")

    # 5 日預測標籤會使用未來 5 個交易日，因此預測第 i 筆時，訓練只取到 i-5 之前。
    start_i = min_train + 5
    prediction_parts: list[pd.DataFrame] = []
    i = start_i
    while i < len(model_df):
        train_end = i - 5
        train = model_df.iloc[:train_end]
        block_end = min(i + retrain_every, len(model_df))
        block = model_df.iloc[i:block_end]
        if len(train) < min_train or block.empty or train["target"].nunique() < 2:
            i = block_end
            continue

        model = _new_model(110)
        model.fit(train[FEATURES], train["target"].astype(int))
        prob = model.predict_proba(block[FEATURES])[:, 1]
        piece = block[["date", "close", "target", "next_day_ret"]].copy()
        piece["prob_up_5d"] = prob
        piece["train_rows_at_prediction"] = len(train)
        prediction_parts.append(piece)
        i = block_end

    if not prediction_parts:
        raise ValueError("walk-forward 無法建立足夠的測試區間。")

    bt = pd.concat(prediction_parts, ignore_index=True).dropna(subset=["prob_up_5d"])
    y_true = bt["target"].astype(int)
    pred = (bt["prob_up_5d"] >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_true, pred))
    auc = float(roc_auc_score(y_true, bt["prob_up_5d"])) if y_true.nunique() == 2 else None
    brier = float(brier_score_loss(y_true, bt["prob_up_5d"]))

    bt["position"] = (bt["prob_up_5d"] >= threshold).astype(int)
    bt["strategy_ret_gross"] = bt["position"] * bt["next_day_ret"].fillna(0)
    bt["benchmark_ret"] = bt["next_day_ret"].fillna(0)
    bt["strategy_curve_gross"] = (1 + bt["strategy_ret_gross"]).cumprod()
    bt["benchmark_curve"] = (1 + bt["benchmark_ret"]).cumprod()

    # 最終模型：所有「未來 5 日結果已知」資料；最新 5 日自然不會進訓練集。
    final_model = _new_model(220)
    final_model.fit(model_df[FEATURES], model_df["target"].astype(int))
    latest = work.dropna(subset=FEATURES).iloc[[-1]]
    probability = float(final_model.predict_proba(latest[FEATURES])[:, 1][0])

    importance = pd.DataFrame(
        {"feature": FEATURES, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return ModelResult(
        probability_up_5d=probability,
        accuracy=accuracy,
        auc=auc,
        brier=brier,
        train_rows=int(bt["train_rows_at_prediction"].max()),
        test_rows=len(bt),
        threshold=threshold,
        retrain_every=retrain_every,
        backtest=bt,
        feature_importance=importance,
    )


def apply_transaction_cost(bt: pd.DataFrame, one_way_cost: float = 0.0015) -> pd.DataFrame:
    out = bt.copy()
    prev_position = out["position"].shift(1).fillna(0)
    out["turnover"] = (out["position"] - prev_position).abs()
    out["cost"] = out["turnover"] * float(one_way_cost)
    out["strategy_ret"] = out["strategy_ret_gross"] - out["cost"]
    out["strategy_curve"] = (1 + out["strategy_ret"]).cumprod()
    return out


def backtest_stats(bt: pd.DataFrame) -> dict[str, float]:
    if bt.empty:
        return {}
    strategy_curve = bt["strategy_curve"] if "strategy_curve" in bt else bt["strategy_curve_gross"]
    trades = int(((bt["position"] == 1) & (bt["position"].shift(1).fillna(0) == 0)).sum())
    invested = bt.loc[bt["position"] == 1, "strategy_ret"] if "strategy_ret" in bt else bt.loc[bt["position"] == 1, "strategy_ret_gross"]
    return {
        "strategy_total": float(strategy_curve.iloc[-1] - 1),
        "benchmark_total": float(bt["benchmark_curve"].iloc[-1] - 1),
        "strategy_mdd": max_drawdown(strategy_curve),
        "trades": float(trades),
        "invested_days": float(bt["position"].sum()),
        "positive_invested_day_ratio": float((invested > 0).mean()) if len(invested) else float("nan"),
    }


def technical_screen_row(stock_id: str, df: pd.DataFrame, stock_name: str = "") -> dict:
    enriched = add_indicators(df)
    clean = enriched.dropna(subset=["strength_score", "ret_20", "volume_ratio", "rsi14"])
    if clean.empty:
        raise ValueError("資料不足")
    latest = clean.iloc[-1]
    prev = clean.iloc[-2] if len(clean) > 1 else latest

    momentum_component = float(np.clip(50 + latest["ret_20"] * 220, 0, 100))
    volume_direction = 1 if latest["ret_1"] >= 0 else -1
    volume_component = float(np.clip(50 + (latest["volume_ratio"] - 1) * 30 * volume_direction, 0, 100))
    location_component = float(np.clip(latest["breakout_pos20"] * 100, 0, 100))
    rank_score = (
        0.60 * float(latest["strength_score"])
        + 0.20 * momentum_component
        + 0.10 * volume_component
        + 0.10 * location_component
    )

    return {
        "代號": normalize_stock_id(stock_id),
        "名稱": stock_name,
        "收盤": float(latest["close"]),
        "日漲跌%": float((latest["close"] / prev["close"] - 1) * 100) if prev["close"] else np.nan,
        "20日動能%": float(latest["ret_20"] * 100),
        "力道": float(latest["strength_score"]),
        "四色狀態": strength_label(float(latest["strength_score"])),
        "RSI": float(latest["rsi14"]),
        "量比": float(latest["volume_ratio"]),
        "V/A": latest["signal"] or "—",
        "技術排名分數": float(rank_score),
        "資料日": pd.Timestamp(latest["date"]).date().isoformat(),
    }


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return float("nan")
    running_max = curve.cummax()
    dd = curve / running_max - 1
    return float(dd.min())



def signal_performance_stats(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Evaluate historical V/A signals without changing how signals are generated.

    For V, a win means price is higher after the selected horizon.
    For A, a win means price is lower after the selected horizon.
    The function is for ex-post evaluation only; future prices are not used to create signals.
    """
    work = df.copy()
    if "signal" not in work.columns or "strength_score" not in work.columns:
        work = add_indicators(work)
    work = work.sort_values("date").reset_index(drop=True)

    rows: list[dict] = []
    for horizon in horizons:
        forward = work["close"].shift(-horizon) / work["close"] - 1
        for signal in ("V", "A"):
            mask = (work["signal"] == signal) & forward.notna()
            sample = forward.loc[mask]
            if sample.empty:
                rows.append({
                    "訊號": signal,
                    "期間": f"{horizon}日",
                    "樣本數": 0,
                    "勝率": np.nan,
                    "平均方向報酬": np.nan,
                    "中位方向報酬": np.nan,
                    "平均原始漲跌": np.nan,
                })
                continue

            directional = sample if signal == "V" else -sample
            rows.append({
                "訊號": signal,
                "期間": f"{horizon}日",
                "樣本數": int(len(sample)),
                "勝率": float((directional > 0).mean()),
                "平均方向報酬": float(directional.mean()),
                "中位方向報酬": float(directional.median()),
                "平均原始漲跌": float(sample.mean()),
            })

    return pd.DataFrame(rows)


def recent_signal_log(
    df: pd.DataFrame,
    horizon: int = 5,
    limit: int = 30,
) -> pd.DataFrame:
    """Return recent V/A occurrences and their later outcome when enough data exists."""
    work = df.copy()
    if "signal" not in work.columns or "strength_score" not in work.columns:
        work = add_indicators(work)
    work = work.sort_values("date").reset_index(drop=True)
    work["future_return"] = work["close"].shift(-horizon) / work["close"] - 1
    signals = work.loc[work["signal"].isin(["V", "A"]), [
        "date", "close", "signal", "strength_score", "future_return"
    ]].copy()
    if signals.empty:
        return pd.DataFrame(columns=["日期", "訊號", "當日收盤", "力道", f"{horizon}日後漲跌%", "方向結果"])

    direction_return = np.where(
        signals["signal"].eq("V"),
        signals["future_return"],
        -signals["future_return"],
    )
    signals["direction_result"] = np.where(
        signals["future_return"].isna(),
        "尚未完成",
        np.where(direction_return > 0, "成功", "失敗"),
    )
    signals["future_pct"] = signals["future_return"] * 100
    out = signals.rename(columns={
        "date": "日期",
        "signal": "訊號",
        "close": "當日收盤",
        "strength_score": "力道",
        "future_pct": f"{horizon}日後漲跌%",
        "direction_result": "方向結果",
    })[["日期", "訊號", "當日收盤", "力道", f"{horizon}日後漲跌%", "方向結果"]]
    return out.tail(limit).sort_values("日期", ascending=False).reset_index(drop=True)


def _directional_signal_sample(
    work: pd.DataFrame,
    signal: str,
    horizon: int,
    before_index: int | None = None,
) -> pd.Series:
    """Completed directional returns for one signal type.

    If before_index is supplied, only signals whose outcome was already observable
    before that row are included. This is used by the contemporaneous V quality score
    to avoid letting the current signal benefit from its own future outcome.
    """
    frame = work.sort_values("date").reset_index(drop=True).copy()
    forward = frame["close"].shift(-horizon) / frame["close"] - 1
    mask = frame["signal"].eq(signal) & forward.notna()
    if before_index is not None:
        # Outcome at i is known only when i + horizon < before_index.
        positions = pd.Series(np.arange(len(frame)), index=frame.index)
        mask &= (positions + int(horizon)) < int(before_index)
    raw = forward.loc[mask]
    return raw if signal == "V" else -raw


def signal_expectancy_stats(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Return win/loss economics for historical V/A signals.

    Returns are converted to a directional basis: positive means the signal direction
    was correct (price up after V, price down after A). This is ex-post research only.
    """
    work = df.copy()
    if "signal" not in work.columns or "strength_score" not in work.columns:
        work = add_indicators(work)
    work = work.sort_values("date").reset_index(drop=True)

    rows: list[dict] = []
    for horizon in horizons:
        for signal in ("V", "A"):
            directional = _directional_signal_sample(work, signal, horizon)
            if directional.empty:
                rows.append({
                    "訊號": signal,
                    "期間": f"{horizon}日",
                    "樣本數": 0,
                    "勝率": np.nan,
                    "期望方向報酬": np.nan,
                    "平均獲利": np.nan,
                    "平均虧損": np.nan,
                    "盈虧比": np.nan,
                    "Profit Factor": np.nan,
                })
                continue

            wins = directional[directional > 0]
            losses = directional[directional < 0]
            avg_win = float(wins.mean()) if len(wins) else np.nan
            avg_loss = float(losses.mean()) if len(losses) else np.nan
            payoff = (
                float(avg_win / abs(avg_loss))
                if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0
                else np.nan
            )
            gross_profit = float(wins.sum()) if len(wins) else 0.0
            gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
            profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.nan

            rows.append({
                "訊號": signal,
                "期間": f"{horizon}日",
                "樣本數": int(len(directional)),
                "勝率": float((directional > 0).mean()),
                "期望方向報酬": float(directional.mean()),
                "平均獲利": avg_win,
                "平均虧損": avg_loss,
                "盈虧比": payoff,
                "Profit Factor": profit_factor,
            })
    return pd.DataFrame(rows)


def _signal_quality_grade(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "—"
    if score >= 80:
        return "A"
    if score >= 68:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def v_signal_quality(
    df: pd.DataFrame,
    signal_index: int | None = None,
    horizon: int = 5,
) -> dict[str, float | int | str]:
    """Explainable quality score for a V signal, using only information available then.

    70 points come from contemporaneous technical conditions. Up to 30 points come
    from *earlier completed* V signals. The current V signal's future return can never
    enter its own quality score.
    """
    work = df.copy()
    if "signal" not in work.columns or "strength_score" not in work.columns:
        work = add_indicators(work)
    work = work.sort_values("date").reset_index(drop=True)

    if signal_index is None:
        idxs = work.index[work["signal"].eq("V")].tolist()
        if not idxs:
            return {"分數": np.nan, "等級": "—", "歷史樣本": 0, "歷史勝率": np.nan, "歷史期望報酬": np.nan}
        signal_index = int(idxs[-1])
    signal_index = int(signal_index)
    if signal_index < 0 or signal_index >= len(work) or work.loc[signal_index, "signal"] != "V":
        return {"分數": np.nan, "等級": "—", "歷史樣本": 0, "歷史勝率": np.nan, "歷史期望報酬": np.nan}

    row = work.loc[signal_index]
    tech = 0.0
    # 25: raw strength above the V trigger, saturated near 80.
    tech += float(np.clip((float(row["strength_score"]) - 55.0) / 25.0, 0, 1)) * 25
    # 15: trend alignment.
    tech += 10 if pd.notna(row["ma20"]) and pd.notna(row["ma60"]) and row["ma20"] > row["ma60"] else 0
    tech += 5 if pd.notna(row["ma20"]) and row["close"] > row["ma20"] else 0
    # 10: momentum quality.
    tech += 5 if pd.notna(row["macd_hist"]) and row["macd_hist"] > 0 else 0
    tech += 5 if pd.notna(row["rsi14"]) and 50 <= row["rsi14"] <= 75 else 0
    # 8: volume participation, capped to avoid extreme-volume distortion.
    volume_ratio = float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else 1.0
    tech += float(np.clip((volume_ratio - 0.8) / 0.8, 0, 1)) * 8
    # 7: location inside the recent 20-day range.
    location = float(row["breakout_pos20"]) if pd.notna(row["breakout_pos20"]) else 0.5
    tech += float(np.clip((location - 0.45) / 0.55, 0, 1)) * 7
    # 5: strength slope known at the signal date.
    if signal_index >= 3:
        delta3 = float(row["strength_score"] - work.loc[signal_index - 3, "strength_score"])
        tech += float(np.clip(delta3 / 15.0, 0, 1)) * 5

    hist = _directional_signal_sample(work, "V", horizon, before_index=signal_index)
    hist_n = int(len(hist))
    if hist_n:
        win_rate = float((hist > 0).mean())
        expectancy = float(hist.mean())
    else:
        win_rate = np.nan
        expectancy = np.nan

    history_points = 0.0
    # Up to 10 points for sample depth, 10 for win rate, 10 for expectancy.
    history_points += min(hist_n / 12.0, 1.0) * 10
    if pd.notna(win_rate):
        history_points += float(np.clip((win_rate - 0.40) / 0.25, 0, 1)) * 10
    if pd.notna(expectancy):
        history_points += float(np.clip((expectancy + 0.005) / 0.035, 0, 1)) * 10

    score = float(np.clip(tech + history_points, 0, 100))
    return {
        "分數": score,
        "等級": _signal_quality_grade(score),
        "技術分": float(tech),
        "歷史分": float(history_points),
        "歷史樣本": hist_n,
        "歷史勝率": win_rate,
        "歷史期望報酬": expectancy,
    }


def _bars_since_latest(work: pd.DataFrame, signal: str) -> float:
    idxs = work.index[work["signal"].eq(signal)].tolist()
    if not idxs:
        return np.nan
    return float(len(work) - 1 - int(idxs[-1]))


def _consecutive_rising_steps(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return 0
    diffs = values.diff().dropna().to_numpy()
    count = 0
    for x in diffs[::-1]:
        if x > 0:
            count += 1
        else:
            break
    return int(count)


def watchlist_signal_row(stock_id: str, df: pd.DataFrame, stock_name: str = "") -> dict:
    """v4 daily signal-center row with recent-V windows and signal economics."""
    enriched = add_indicators(df).sort_values("date").reset_index(drop=True)
    clean = enriched.dropna(subset=["strength_score", "ret_20", "volume_ratio", "rsi14"]).reset_index(drop=True)
    if len(clean) < 8:
        raise ValueError("資料不足")

    latest = clean.iloc[-1]
    prev = clean.iloc[-2]
    signal_rows = clean.loc[clean["signal"].isin(["V", "A"])].copy()

    if signal_rows.empty:
        last_signal = "—"
        last_signal_date = "—"
        bars_since = np.nan
    else:
        last_idx = int(signal_rows.index[-1])
        last_row = signal_rows.iloc[-1]
        last_signal = str(last_row["signal"])
        last_signal_date = pd.Timestamp(last_row["date"]).date().isoformat()
        bars_since = int(len(clean) - 1 - last_idx)

    bars_v = _bars_since_latest(clean, "V")
    bars_a = _bars_since_latest(clean, "A")
    delta3 = float(latest["strength_score"] - clean.iloc[-4]["strength_score"])
    delta5 = float(latest["strength_score"] - clean.iloc[-6]["strength_score"])
    rising_steps = _consecutive_rising_steps(clean["strength_score"])
    rapid_warming = bool(delta3 >= 10 or delta5 >= 15)

    momentum_component = float(np.clip(50 + latest["ret_20"] * 220, 0, 100))
    volume_direction = 1 if latest["ret_1"] >= 0 else -1
    volume_component = float(np.clip(50 + (latest["volume_ratio"] - 1) * 30 * volume_direction, 0, 100))
    location_component = float(np.clip(latest["breakout_pos20"] * 100, 0, 100))
    rank_score = (
        0.60 * float(latest["strength_score"])
        + 0.20 * momentum_component
        + 0.10 * volume_component
        + 0.10 * location_component
    )

    stats = signal_performance_stats(enriched, horizons=(5,))
    economics = signal_expectancy_stats(enriched, horizons=(5,))

    def _extract_basic(signal: str) -> tuple[float, int]:
        row = stats[(stats["訊號"] == signal) & (stats["期間"] == "5日")]
        if row.empty:
            return np.nan, 0
        r = row.iloc[0]
        return float(r["勝率"]) if pd.notna(r["勝率"]) else np.nan, int(r["樣本數"])

    def _extract_econ(signal: str) -> tuple[float, float, float]:
        row = economics[(economics["訊號"] == signal) & (economics["期間"] == "5日")]
        if row.empty:
            return np.nan, np.nan, np.nan
        r = row.iloc[0]
        return (
            float(r["期望方向報酬"]) if pd.notna(r["期望方向報酬"]) else np.nan,
            float(r["盈虧比"]) if pd.notna(r["盈虧比"]) else np.nan,
            float(r["Profit Factor"]) if pd.notna(r["Profit Factor"]) else np.nan,
        )

    v_win, v_n = _extract_basic("V")
    a_win, a_n = _extract_basic("A")
    v_exp, v_payoff, v_pf = _extract_econ("V")
    a_exp, a_payoff, a_pf = _extract_econ("A")
    fresh = str(latest["signal"]) if latest["signal"] in ("V", "A") else "—"

    if pd.notna(bars_v):
        v_idx = len(clean) - 1 - int(bars_v)
        v_quality = v_signal_quality(clean, signal_index=v_idx, horizon=5)
    else:
        v_quality = {"分數": np.nan, "等級": "—", "歷史樣本": 0, "歷史勝率": np.nan, "歷史期望報酬": np.nan}

    return {
        "代號": normalize_stock_id(stock_id),
        "名稱": stock_name,
        "資料日": pd.Timestamp(latest["date"]).date().isoformat(),
        "收盤": float(latest["close"]),
        "日漲跌%": float((latest["close"] / prev["close"] - 1) * 100) if prev["close"] else np.nan,
        "力道": float(latest["strength_score"]),
        "四色狀態": strength_label(float(latest["strength_score"])),
        "力道3日變化": delta3,
        "力道5日變化": delta5,
        "連續轉強日數": rising_steps,
        "快速升溫": "是" if rapid_warming else "—",
        "最新資料日V/A": fresh,
        "近1日新V": "是" if pd.notna(bars_v) and bars_v <= 0 else "—",
        "近3日新V": "是" if pd.notna(bars_v) and bars_v <= 2 else "—",
        "近5日新V": "是" if pd.notna(bars_v) and bars_v <= 4 else "—",
        "近1日新A": "是" if pd.notna(bars_a) and bars_a <= 0 else "—",
        "近3日新A": "是" if pd.notna(bars_a) and bars_a <= 2 else "—",
        "近5日新A": "是" if pd.notna(bars_a) and bars_a <= 4 else "—",
        "最近V距今交易日": int(bars_v) if pd.notna(bars_v) else np.nan,
        "最近V品質": str(v_quality["等級"]),
        "V品質分數": float(v_quality["分數"]) if pd.notna(v_quality["分數"]) else np.nan,
        "上次V/A": last_signal,
        "上次訊號日": last_signal_date,
        "距上次訊號交易日": bars_since,
        "20日動能%": float(latest["ret_20"] * 100),
        "RSI": float(latest["rsi14"]),
        "量比": float(latest["volume_ratio"]),
        "V後5日勝率%": float(v_win * 100) if pd.notna(v_win) else np.nan,
        "V樣本": int(v_n),
        "V後5日期望報酬%": float(v_exp * 100) if pd.notna(v_exp) else np.nan,
        "V盈虧比": v_payoff,
        "V Profit Factor": v_pf,
        "A後5日勝率%": float(a_win * 100) if pd.notna(a_win) else np.nan,
        "A樣本": int(a_n),
        "A後5日期望報酬%": float(a_exp * 100) if pd.notna(a_exp) else np.nan,
        "A盈虧比": a_payoff,
        "A Profit Factor": a_pf,
        "技術排名分數": float(rank_score),
    }


def strategy_lab_backtest(
    df: pd.DataFrame,
    allowed_grades: tuple[str, ...] = ("A", "B"),
    min_strength: float = 55.0,
    min_volume_ratio: float = 1.0,
    min_momentum_20: float | None = None,
    hold_days: int = 10,
    stop_loss_pct: float | None = 0.06,
    take_profit_pct: float | None = 0.12,
    one_way_cost: float = 0.0015,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Backtest a configurable long-only V-signal strategy without look-ahead.

    Rules:
    - A V signal is known only after that day's close.
    - Entry is therefore the *next trading day's open*.
    - Signal filters use only values available on the V-signal date.
    - V quality uses only contemporaneous information plus earlier completed V samples.
    - Only one full-capital position may be open at a time; overlapping V signals are ignored.
    - Stop/take levels are fixed from entry. If both are touched in one daily bar, the
      conservative assumption is that the stop is hit first.
    - If no stop/take is hit, exit at the close of the selected holding-period day.
    """
    if hold_days < 1:
        raise ValueError("持有天數至少要 1 個交易日。")
    if one_way_cost < 0:
        raise ValueError("交易成本不可為負數。")
    if not allowed_grades:
        raise ValueError("請至少選一個 V 品質等級。")

    valid_grades = {"A", "B", "C", "D"}
    selected_grades = tuple(g for g in allowed_grades if g in valid_grades)
    if not selected_grades:
        raise ValueError("V 品質等級必須為 A、B、C 或 D。")

    work = df.copy()
    if "signal" not in work.columns or "strength_score" not in work.columns:
        work = add_indicators(work)
    work = work.sort_values("date").reset_index(drop=True)

    required = [
        "date", "open", "high", "low", "close", "signal", "strength_score",
        "volume_ratio", "ret_20",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"策略研究缺少欄位：{', '.join(missing)}")

    trade_rows: list[dict] = []
    qualifying_signals = 0
    skipped_overlap = 0
    skipped_incomplete = 0
    last_exit_idx = -1

    signal_indices = work.index[work["signal"].eq("V")].tolist()
    for signal_idx in signal_indices:
        signal_idx = int(signal_idx)
        # Need a next-day open to avoid entering before the signal is actually known.
        if signal_idx + 1 >= len(work):
            continue

        row = work.loc[signal_idx]
        quality = v_signal_quality(work, signal_index=signal_idx, horizon=5)
        grade = str(quality.get("等級", "—"))

        if grade not in selected_grades:
            continue
        if pd.isna(row["strength_score"]) or float(row["strength_score"]) < float(min_strength):
            continue
        if pd.isna(row["volume_ratio"]) or float(row["volume_ratio"]) < float(min_volume_ratio):
            continue
        if min_momentum_20 is not None:
            if pd.isna(row["ret_20"]) or float(row["ret_20"]) * 100 < float(min_momentum_20):
                continue

        qualifying_signals += 1
        entry_idx = signal_idx + 1
        # A completed backtest trade must have the full selected holding window available.
        # Recent signals without enough future bars are excluded rather than force-closed at the data end.
        if entry_idx + int(hold_days) - 1 >= len(work):
            skipped_incomplete += 1
            continue
        if entry_idx <= last_exit_idx:
            skipped_overlap += 1
            continue

        entry_row = work.loc[entry_idx]
        entry_price = float(entry_row["open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        stop_price = None
        take_price = None
        if stop_loss_pct is not None and float(stop_loss_pct) > 0:
            stop_price = entry_price * (1 - float(stop_loss_pct))
        if take_profit_pct is not None and float(take_profit_pct) > 0:
            take_price = entry_price * (1 + float(take_profit_pct))

        max_exit_idx = entry_idx + int(hold_days) - 1
        exit_idx = max_exit_idx
        exit_price = float(work.loc[max_exit_idx, "close"])
        exit_reason = f"持有{int(hold_days)}日"

        for j in range(entry_idx, max_exit_idx + 1):
            day = work.loc[j]
            day_open = float(day["open"])
            day_high = float(day["high"])
            day_low = float(day["low"])

            # Gap handling comes first. A gap beyond a level is filled at the open,
            # not at an impossible better stop/take price.
            if stop_price is not None and day_open <= stop_price:
                exit_idx = j
                exit_price = day_open
                exit_reason = "停損（跳空）"
                break
            if take_price is not None and day_open >= take_price:
                exit_idx = j
                exit_price = day_open
                exit_reason = "停利（跳空）"
                break

            hit_stop = stop_price is not None and day_low <= stop_price
            hit_take = take_price is not None and day_high >= take_price

            if hit_stop and hit_take:
                exit_idx = j
                exit_price = float(stop_price)
                exit_reason = "同日觸發停損/停利→保守採停損"
                break
            if hit_stop:
                exit_idx = j
                exit_price = float(stop_price)
                exit_reason = "停損"
                break
            if hit_take:
                exit_idx = j
                exit_price = float(take_price)
                exit_reason = "停利"
                break

        # Transaction-cost model: cost is paid on both entry and exit notionals.
        gross_return = exit_price / entry_price - 1
        net_return = (exit_price * (1 - float(one_way_cost))) / (
            entry_price * (1 + float(one_way_cost))
        ) - 1

        trade_rows.append({
            "訊號日": pd.Timestamp(row["date"]),
            "進場日": pd.Timestamp(entry_row["date"]),
            "出場日": pd.Timestamp(work.loc[exit_idx, "date"]),
            "V品質": grade,
            "V品質分數": float(quality["分數"]) if pd.notna(quality.get("分數")) else np.nan,
            "訊號力道": float(row["strength_score"]),
            "訊號量比": float(row["volume_ratio"]),
            "訊號20日動能%": float(row["ret_20"] * 100) if pd.notna(row["ret_20"]) else np.nan,
            "進場價": entry_price,
            "出場價": exit_price,
            "毛報酬": gross_return,
            "淨報酬": net_return,
            "持有交易日": int(exit_idx - entry_idx + 1),
            "出場原因": exit_reason,
            "_entry_idx": int(entry_idx),
            "_exit_idx": int(exit_idx),
        })
        last_exit_idx = int(exit_idx)

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        equity = pd.DataFrame(columns=["date", "equity"])
        return trades, equity, {
            "qualifying_signals": float(qualifying_signals),
            "trades": 0.0,
            "skipped_overlap": float(skipped_overlap),
            "skipped_incomplete": float(skipped_incomplete),
            "win_rate": np.nan,
            "expectancy": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "payoff_ratio": np.nan,
            "profit_factor": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "best_trade": np.nan,
            "worst_trade": np.nan,
            "avg_holding_days": np.nan,
        }

    net = pd.to_numeric(trades["淨報酬"], errors="coerce").dropna()
    wins = net[net > 0]
    losses = net[net < 0]
    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(losses.mean()) if len(losses) else np.nan
    payoff_ratio = (
        float(avg_win / abs(avg_loss))
        if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0
        else np.nan
    )
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.nan

    # Daily mark-to-market equity curve. While flat, capital is unchanged; while in a
    # trade, the curve follows daily closes and uses the actual simulated exit price
    # on the exit day. This makes drawdown more informative than an exit-only curve.
    capital = 1.0
    equity_rows: list[dict] = []
    first_entry_idx = int(trades["_entry_idx"].min())
    last_trade_exit_idx = int(trades["_exit_idx"].max())
    if first_entry_idx > 0:
        equity_rows.append({"date": pd.Timestamp(work.loc[first_entry_idx - 1, "date"]), "equity": 1.0})

    trade_records = trades.sort_values("_entry_idx").to_dict("records")
    tpos = 0
    current = None
    shares = 0.0
    for j in range(first_entry_idx, last_trade_exit_idx + 1):
        if current is None and tpos < len(trade_records) and j == int(trade_records[tpos]["_entry_idx"]):
            current = trade_records[tpos]
            entry_price = float(current["進場價"])
            shares = capital / (entry_price * (1 + float(one_way_cost)))

        if current is None:
            value = capital
        elif j == int(current["_exit_idx"]):
            value = shares * float(current["出場價"]) * (1 - float(one_way_cost))
            capital = value
            current = None
            shares = 0.0
            tpos += 1
        else:
            value = shares * float(work.loc[j, "close"])

        equity_rows.append({"date": pd.Timestamp(work.loc[j, "date"]), "equity": float(value)})

    equity = pd.DataFrame(equity_rows)
    total_return = float(equity["equity"].iloc[-1] - 1) if not equity.empty else 0.0
    stats = {
        "qualifying_signals": float(qualifying_signals),
        "trades": float(len(trades)),
        "skipped_overlap": float(skipped_overlap),
        "skipped_incomplete": float(skipped_incomplete),
        "win_rate": float((net > 0).mean()) if len(net) else np.nan,
        "expectancy": float(net.mean()) if len(net) else np.nan,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "total_return": total_return,
        "max_drawdown": max_drawdown(equity["equity"]) if not equity.empty else 0.0,
        "best_trade": float(net.max()) if len(net) else np.nan,
        "worst_trade": float(net.min()) if len(net) else np.nan,
        "avg_holding_days": float(pd.to_numeric(trades["持有交易日"], errors="coerce").mean()),
    }
    trades = trades.drop(columns=["_entry_idx", "_exit_idx"], errors="ignore")
    return trades, equity, stats

# -----------------------------------------------------------------------------
# v6 forward validation / paper tracking
# -----------------------------------------------------------------------------
MODEL_VERSION = "v6-model-1.0"
FORWARD_HORIZONS = (5, 10, 20)


def latest_prediction_probability(df: pd.DataFrame, min_train: int = 260) -> tuple[float, int]:
    """Fit only on samples whose 5-day outcome is already known, then score latest row.

    This is intentionally lighter than the full walk-forward report and is used by the
    scheduled forward tracker. The recorded probability is frozen in the snapshot.
    """
    work = df.copy()
    if "strength_score" not in work.columns:
        work = add_indicators(work)
    work = work.sort_values("date").reset_index(drop=True)
    work["target_ret_5"] = work["close"].shift(-5) / work["close"] - 1
    work["target"] = np.where(
        work["target_ret_5"].notna(),
        (work["target_ret_5"] > 0).astype(int),
        np.nan,
    )

    train = work.dropna(subset=FEATURES + ["target"]).copy()
    latest = work.dropna(subset=FEATURES).tail(1)
    if len(train) < int(min_train) or latest.empty:
        return np.nan, int(len(train))
    y = train["target"].astype(int)
    if y.nunique() < 2:
        return float(y.iloc[-1]), int(len(train))

    model = _new_model(220)
    model.fit(train[FEATURES], y)
    probability = float(model.predict_proba(latest[FEATURES])[:, 1][0])
    return probability, int(len(train))


def forward_snapshot_row(
    stock_id: str,
    df: pd.DataFrame,
    stock_name: str = "",
    source: str = "",
    model_version: str = MODEL_VERSION,
    recorded_at: str | None = None,
) -> dict:
    """Create an immutable-as-of-now daily snapshot for forward validation."""
    work = df.copy()
    if "strength_score" not in work.columns:
        work = add_indicators(work)
    work = work.sort_values("date").reset_index(drop=True)
    clean = work.dropna(subset=FEATURES + ["close", "rsi14", "volume_ratio", "strength_score"])
    if clean.empty:
        raise ValueError("資料不足，無法建立前向驗證快照。")

    latest_idx = int(clean.index[-1])
    latest = work.loc[latest_idx]
    probability, train_rows = latest_prediction_probability(work)
    signal = str(latest.get("signal", "")) if str(latest.get("signal", "")) in ("V", "A") else ""

    quality = {
        "分數": np.nan,
        "等級": "—",
        "歷史樣本": 0,
        "歷史勝率": np.nan,
        "歷史期望報酬": np.nan,
    }
    if signal == "V":
        quality = v_signal_quality(work, signal_index=latest_idx, horizon=5)

    if recorded_at is None:
        recorded_at = pd.Timestamp.now(tz="Asia/Taipei").isoformat()

    return {
        "recorded_at": recorded_at,
        "data_date": pd.Timestamp(latest["date"]).date().isoformat(),
        "stock_id": normalize_stock_id(stock_id),
        "stock_name": stock_name,
        "model_version": model_version,
        "source": source,
        "close": float(latest["close"]),
        "signal": signal,
        "v_grade": str(quality.get("等級", "—")) if signal == "V" else "—",
        "v_quality_score": float(quality.get("分數")) if signal == "V" and pd.notna(quality.get("分數")) else np.nan,
        "strength_score": float(latest["strength_score"]),
        "strength_band": strength_label(float(latest["strength_score"])),
        "rsi14": float(latest["rsi14"]),
        "volume_ratio": float(latest["volume_ratio"]),
        "ret_20_pct": float(latest["ret_20"] * 100) if pd.notna(latest["ret_20"]) else np.nan,
        "ai_up_5d_prob": probability,
        "model_train_rows": train_rows,
        # Future-result fields are intentionally empty at snapshot time.
        "ret_5d": np.nan,
        "ret_10d": np.nan,
        "ret_20d": np.nan,
        "mfe_5d": np.nan,
        "mae_5d": np.nan,
        "mfe_10d": np.nan,
        "mae_10d": np.nan,
        "mfe_20d": np.nan,
        "mae_20d": np.nan,
        "benchmark_5d": np.nan,
        "benchmark_10d": np.nan,
        "benchmark_20d": np.nan,
        "excess_5d": np.nan,
        "excess_10d": np.nan,
        "excess_20d": np.nan,
        "evaluated_5d_at": "",
        "evaluated_10d_at": "",
        "evaluated_20d_at": "",
    }


def _date_position(work: pd.DataFrame, date_value: str | pd.Timestamp) -> int | None:
    dates = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    target = pd.Timestamp(date_value).normalize()
    hits = np.flatnonzero(dates.eq(target).to_numpy())
    return int(hits[-1]) if len(hits) else None


def update_forward_outcomes(
    log: pd.DataFrame,
    stock_id: str,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> pd.DataFrame:
    """Fill only previously-empty future outcome cells; snapshot inputs stay untouched."""
    if log.empty:
        return log.copy()

    out = log.copy()
    work = df.copy()
    if "strength_score" not in work.columns:
        work = add_indicators(work)
    work = work.sort_values("date").reset_index(drop=True)

    benchmark = None
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark = benchmark_df.sort_values("date").reset_index(drop=True).copy()

    # CSV readers can infer all-empty evaluated_* columns as float; force object before
    # later writing ISO date strings so future pandas versions do not reject the assignment.
    for horizon in horizons:
        eval_col = f"evaluated_{int(horizon)}d_at"
        if eval_col in out.columns:
            out[eval_col] = out[eval_col].astype("object")

    sid = normalize_stock_id(stock_id)
    ticker_rows = out.index[out["stock_id"].astype(str).eq(sid)].tolist()
    for row_idx in ticker_rows:
        base_pos = _date_position(work, out.at[row_idx, "data_date"])
        if base_pos is None:
            continue
        base_close = float(work.loc[base_pos, "close"])
        if not np.isfinite(base_close) or base_close <= 0:
            continue

        benchmark_pos = None
        benchmark_base = None
        if benchmark is not None:
            benchmark_pos = _date_position(benchmark, out.at[row_idx, "data_date"])
            if benchmark_pos is not None:
                benchmark_base = float(benchmark.loc[benchmark_pos, "close"])

        for horizon in horizons:
            ret_col = f"ret_{horizon}d"
            mfe_col = f"mfe_{horizon}d"
            mae_col = f"mae_{horizon}d"
            bench_col = f"benchmark_{horizon}d"
            excess_col = f"excess_{horizon}d"
            eval_col = f"evaluated_{horizon}d_at"

            # Never rewrite an already-realized outcome. This preserves the first
            # forward observation even if a data vendor later revises history.
            existing = pd.to_numeric(pd.Series([out.at[row_idx, ret_col]]), errors="coerce").iloc[0]
            if pd.notna(existing):
                continue

            end_pos = base_pos + int(horizon)
            if end_pos >= len(work):
                continue

            future = work.iloc[base_pos + 1 : end_pos + 1]
            end_close = float(work.loc[end_pos, "close"])
            realized = end_close / base_close - 1
            mfe = float(future["high"].max() / base_close - 1) if not future.empty else np.nan
            mae = float(future["low"].min() / base_close - 1) if not future.empty else np.nan

            out.at[row_idx, ret_col] = realized
            out.at[row_idx, mfe_col] = mfe
            out.at[row_idx, mae_col] = mae
            out.at[row_idx, eval_col] = pd.Timestamp(work.loc[end_pos, "date"]).date().isoformat()

            if benchmark is not None and benchmark_pos is not None and benchmark_base and benchmark_base > 0:
                benchmark_end = benchmark_pos + int(horizon)
                if benchmark_end < len(benchmark):
                    benchmark_ret = float(benchmark.loc[benchmark_end, "close"] / benchmark_base - 1)
                    out.at[row_idx, bench_col] = benchmark_ret
                    out.at[row_idx, excess_col] = realized - benchmark_ret

    return out


def forward_performance_summary(
    log: pd.DataFrame,
    signal: str = "V",
    model_version: str | None = None,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> pd.DataFrame:
    """Summarize immutable forward snapshots after outcomes become available."""
    if log is None or log.empty:
        return pd.DataFrame()
    work = log.copy()
    if model_version:
        work = work[work["model_version"].astype(str).eq(str(model_version))]
    if signal in ("V", "A"):
        work = work[work["signal"].astype(str).eq(signal)]
    if work.empty:
        return pd.DataFrame()

    direction = 1.0 if signal != "A" else -1.0
    rows = []
    for horizon in horizons:
        ret_col = f"ret_{horizon}d"
        bench_col = f"benchmark_{horizon}d"
        excess_col = f"excess_{horizon}d"
        mfe_col = f"mfe_{horizon}d"
        mae_col = f"mae_{horizon}d"
        if ret_col not in work.columns:
            continue
        ret = pd.to_numeric(work[ret_col], errors="coerce")
        mask = ret.notna()
        if not mask.any():
            rows.append({
                "期間": f"{horizon}日", "樣本數": 0, "方向勝率": np.nan,
                "平均方向報酬": np.nan, "中位方向報酬": np.nan,
                "平均相對0050超額": np.nan, "平均MFE": np.nan, "平均MAE": np.nan,
            })
            continue
        directed = ret[mask] * direction
        excess = pd.to_numeric(work.loc[mask, excess_col], errors="coerce") * direction if excess_col in work else pd.Series(dtype=float)
        mfe = pd.to_numeric(work.loc[mask, mfe_col], errors="coerce") if mfe_col in work else pd.Series(dtype=float)
        mae = pd.to_numeric(work.loc[mask, mae_col], errors="coerce") if mae_col in work else pd.Series(dtype=float)
        rows.append({
            "期間": f"{horizon}日",
            "樣本數": int(mask.sum()),
            "方向勝率": float((directed > 0).mean()),
            "平均方向報酬": float(directed.mean()),
            "中位方向報酬": float(directed.median()),
            "平均相對0050超額": float(excess.mean()) if len(excess.dropna()) else np.nan,
            "平均MFE": float(mfe.mean()) if len(mfe.dropna()) else np.nan,
            "平均MAE": float(mae.mean()) if len(mae.dropna()) else np.nan,
        })
    return pd.DataFrame(rows)


def forward_ai_calibration(log: pd.DataFrame, model_version: str | None = None) -> pd.DataFrame:
    """Compare recorded 5-day up probabilities with later realized 5-day outcomes."""
    if log is None or log.empty:
        return pd.DataFrame()
    work = log.copy()
    if model_version:
        work = work[work["model_version"].astype(str).eq(str(model_version))]
    work["prob"] = pd.to_numeric(work.get("ai_up_5d_prob"), errors="coerce")
    work["ret"] = pd.to_numeric(work.get("ret_5d"), errors="coerce")
    work = work.dropna(subset=["prob", "ret"])
    if work.empty:
        return pd.DataFrame()

    bins = [-0.001, 0.40, 0.50, 0.60, 0.70, 1.001]
    labels = ["<40%", "40–49%", "50–59%", "60–69%", "≥70%"]
    work["機率區間"] = pd.cut(work["prob"], bins=bins, labels=labels, include_lowest=True, right=False)
    grouped = work.groupby("機率區間", observed=False)
    rows = []
    for label, group in grouped:
        if group.empty:
            continue
        rows.append({
            "AI機率區間": str(label),
            "樣本數": int(len(group)),
            "平均預測機率": float(group["prob"].mean()),
            "實際5日上漲率": float((group["ret"] > 0).mean()),
            "平均5日報酬": float(group["ret"].mean()),
        })
    return pd.DataFrame(rows)
