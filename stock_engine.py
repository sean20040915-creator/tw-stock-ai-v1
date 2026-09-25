from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


@dataclass
class ModelResult:
    probability_up_5d: float
    accuracy: float
    auc: Optional[float]
    train_rows: int
    test_rows: int
    threshold: float
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
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["volume"] = out["volume"].fillna(0)
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

    # yfinance 的 end 是不含當日，因此加一天。
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

            data = data.reset_index()
            rename_map = {
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
            data = data.rename(columns=rename_map)
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
    start = end - timedelta(days=int(years * 365.25) + 120)
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["ma5"] = close.rolling(5).mean()
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["ma20_ratio"] = close / out["ma20"] - 1
    out["ma60_ratio"] = close / out["ma60"] - 1

    # RSI (Wilder-like smoothing)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))
    out.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100

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

    # 自訂多空力道：只使用當日與過去資料，分數限制 0~100。
    score = pd.Series(50.0, index=out.index)
    score += np.where(close > out["ma20"], 12, -12)
    score += np.where(out["ma20"] > out["ma60"], 10, -10)
    score += np.where(out["macd_hist"] > 0, 8, -8)
    score += np.where(out["rsi14"] >= 55, 8, np.where(out["rsi14"] <= 45, -8, 0))
    score += np.where(out["ret_5"] > 0, 6, -6)
    score += np.where(
        (out["volume_ratio"] > 1.1) & (out["ret_1"] > 0),
        6,
        np.where((out["volume_ratio"] > 1.1) & (out["ret_1"] < 0), -6, 0),
    )
    out["strength_score"] = score.clip(0, 100)

    prev_score = out["strength_score"].shift(1)
    out["signal"] = ""
    out.loc[(out["strength_score"] >= 60) & (prev_score < 60), "signal"] = "V"
    out.loc[(out["strength_score"] <= 40) & (prev_score > 40), "signal"] = "A"

    return out


def strength_label(score: float) -> str:
    if score >= 75:
        return "強多"
    if score >= 60:
        return "偏多"
    if score > 40:
        return "盤整"
    if score > 25:
        return "偏空"
    return "強空"


def recent_support_resistance(df: pd.DataFrame, window: int = 20) -> tuple[float, float]:
    tail = df.tail(window)
    return float(tail["low"].min()), float(tail["high"].max())


def train_prediction_model(df: pd.DataFrame, threshold: float = 0.55) -> ModelResult:
    work = df.copy()
    work["target_ret_5"] = work["close"].shift(-5) / work["close"] - 1
    work["target"] = np.where(
        work["target_ret_5"].notna(),
        (work["target_ret_5"] > 0).astype(int),
        np.nan,
    )
    work["next_day_ret"] = work["close"].pct_change().shift(-1)

    features = [
        "ret_1",
        "ret_5",
        "ma20_ratio",
        "ma60_ratio",
        "rsi14",
        "macd_hist",
        "atr_pct",
        "volatility20",
        "volume_ratio",
        "range_pct",
        "strength_score",
    ]

    model_df = work.dropna(subset=features + ["target"]).copy()
    if len(model_df) < 320:
        raise ValueError("歷史資料不足以建立第一版模型；請把資料期間調成 3 年以上。")

    split = int(len(model_df) * 0.75)
    split = min(max(split, 220), len(model_df) - 60)
    train = model_df.iloc[:split]
    test = model_df.iloc[split:]

    X_train = train[features]
    y_train = train["target"].astype(int)
    X_test = test[features]
    y_test = test["target"].astype(int)

    model = RandomForestClassifier(
        n_estimators=260,
        max_depth=7,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, test_pred))
    auc = None
    if y_test.nunique() == 2:
        auc = float(roc_auc_score(y_test, test_prob))

    # 最終模型：用所有「已知未來 5 日結果」的樣本重新訓練，再預測最新一日。
    final_model = RandomForestClassifier(
        n_estimators=320,
        max_depth=7,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(model_df[features], model_df["target"].astype(int))

    latest = work.dropna(subset=features).iloc[[-1]]
    probability = float(final_model.predict_proba(latest[features])[:, 1][0])

    bt = test[["date", "close", "next_day_ret"]].copy()
    bt["prob_up_5d"] = test_prob
    bt["position"] = (bt["prob_up_5d"] >= threshold).astype(int)
    bt["strategy_ret"] = bt["position"] * bt["next_day_ret"].fillna(0)
    bt["benchmark_ret"] = bt["next_day_ret"].fillna(0)
    bt["strategy_curve"] = (1 + bt["strategy_ret"]).cumprod()
    bt["benchmark_curve"] = (1 + bt["benchmark_ret"]).cumprod()

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return ModelResult(
        probability_up_5d=probability,
        accuracy=acc,
        auc=auc,
        train_rows=len(train),
        test_rows=len(test),
        threshold=threshold,
        backtest=bt,
        feature_importance=importance,
    )


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return float("nan")
    running_max = curve.cummax()
    dd = curve / running_max - 1
    return float(dd.min())
