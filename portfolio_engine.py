from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from stock_engine import add_indicators, v_signal_quality


GRADE_RANK = {"A": 4, "B": 3, "C": 2, "D": 1}


@dataclass
class _Position:
    stock_id: str
    stock_name: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_idx: int
    planned_exit_idx: int
    shares: float
    entry_price: float
    entry_notional: float
    stop_price: float | None
    take_price: float | None
    grade: str
    quality_score: float
    signal_strength: float
    signal_volume_ratio: float
    signal_momentum20_pct: float
    last_mark_price: float


def _max_drawdown(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min())


def _curve_stats(curve: pd.DataFrame, value_col: str = "equity") -> dict[str, float]:
    if curve.empty or value_col not in curve.columns:
        return {
            "total_return": 0.0,
            "cagr": np.nan,
            "max_drawdown": 0.0,
            "sharpe": np.nan,
        }
    values = pd.to_numeric(curve[value_col], errors="coerce").dropna()
    if len(values) < 2 or float(values.iloc[0]) <= 0:
        return {
            "total_return": 0.0,
            "cagr": np.nan,
            "max_drawdown": _max_drawdown(values),
            "sharpe": np.nan,
        }
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    periods = max(len(values) - 1, 1)
    cagr = float((values.iloc[-1] / values.iloc[0]) ** (252.0 / periods) - 1.0)
    daily = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily) >= 2 and float(daily.std(ddof=1)) > 0:
        sharpe = float(np.sqrt(252.0) * daily.mean() / daily.std(ddof=1))
    else:
        sharpe = np.nan
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": _max_drawdown(values),
        "sharpe": sharpe,
    }


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "signal" not in work.columns or "strength_score" not in work.columns:
        work = add_indicators(work)
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    work = work.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return work


def _build_candidates(
    data_map: Dict[str, pd.DataFrame],
    stock_names: dict[str, str],
    allowed_grades: tuple[str, ...],
    min_strength: float,
    min_volume_ratio: float,
    min_momentum_20: float | None,
    hold_days: int,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    frames: dict[str, pd.DataFrame] = {}
    candidates: list[dict] = []

    for stock_id, raw in data_map.items():
        if raw is None or len(raw) < 80:
            continue
        work = _prepare_frame(raw)
        if work.empty:
            continue
        frames[str(stock_id)] = work

        for signal_idx in work.index[work["signal"].eq("V")].tolist():
            signal_idx = int(signal_idx)
            entry_idx = signal_idx + 1
            planned_exit_idx = entry_idx + int(hold_days) - 1
            if entry_idx >= len(work) or planned_exit_idx >= len(work):
                continue

            row = work.loc[signal_idx]
            quality = v_signal_quality(work, signal_index=signal_idx, horizon=5)
            grade = str(quality.get("等級", "—"))
            quality_score = float(quality.get("分數")) if pd.notna(quality.get("分數")) else 0.0

            if grade not in allowed_grades:
                continue
            if pd.isna(row.get("strength_score")) or float(row["strength_score"]) < float(min_strength):
                continue
            if pd.isna(row.get("volume_ratio")) or float(row["volume_ratio"]) < float(min_volume_ratio):
                continue
            momentum = float(row["ret_20"] * 100) if pd.notna(row.get("ret_20")) else np.nan
            if min_momentum_20 is not None and (pd.isna(momentum) or momentum < float(min_momentum_20)):
                continue

            candidates.append({
                "stock_id": str(stock_id),
                "stock_name": str(stock_names.get(str(stock_id), "")),
                "signal_date": pd.Timestamp(row["date"]),
                "entry_date": pd.Timestamp(work.loc[entry_idx, "date"]),
                "signal_idx": signal_idx,
                "entry_idx": entry_idx,
                "planned_exit_idx": planned_exit_idx,
                "grade": grade,
                "grade_rank": GRADE_RANK.get(grade, 0),
                "quality_score": quality_score,
                "strength": float(row["strength_score"]),
                "volume_ratio": float(row["volume_ratio"]),
                "momentum20_pct": momentum,
            })

    candidates.sort(
        key=lambda x: (
            x["entry_date"],
            -x["grade_rank"],
            -x["quality_score"],
            -x["strength"],
            -x["volume_ratio"],
            x["stock_id"],
        )
    )
    return frames, candidates


def portfolio_backtest(
    data_map: Dict[str, pd.DataFrame],
    stock_names: dict[str, str] | None = None,
    initial_capital: float = 1_000_000.0,
    allowed_grades: tuple[str, ...] = ("A", "B"),
    min_strength: float = 55.0,
    min_volume_ratio: float = 1.0,
    min_momentum_20: float | None = None,
    hold_days: int = 10,
    stop_loss_pct: float | None = 0.06,
    take_profit_pct: float | None = 0.12,
    one_way_cost: float = 0.0015,
    max_positions: int = 5,
    position_pct: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Multi-stock, long-only V-signal portfolio simulation.

    Anti-look-ahead rules:
    - V is only known after the signal day's close; entry is next trading day's open.
    - V quality only uses information available at the signal date plus earlier completed samples.
    - If stop and take-profit are both touched in one daily bar, stop is assumed first.
    - A position planned to exit at the close still occupies a slot at that day's open.
      Exit proceeds are not reused for same-day new entries. This is intentionally conservative.
    - No leverage: a new position is capped by available cash even if position_pct * max_positions > 100%.
    """
    if initial_capital <= 0:
        raise ValueError("初始資金必須大於 0。")
    if hold_days < 1:
        raise ValueError("持有天數至少要 1 個交易日。")
    if max_positions < 1:
        raise ValueError("最多同時持股數至少為 1。")
    if not (0 < position_pct <= 1):
        raise ValueError("單檔資金比例必須介於 0% 與 100% 之間。")
    if one_way_cost < 0:
        raise ValueError("交易成本不可為負數。")
    valid_grades = tuple(g for g in allowed_grades if g in GRADE_RANK)
    if not valid_grades:
        raise ValueError("請至少選一個 V 品質等級。")

    stock_names = stock_names or {}
    frames, candidates = _build_candidates(
        data_map,
        stock_names,
        valid_grades,
        float(min_strength),
        float(min_volume_ratio),
        min_momentum_20,
        int(hold_days),
    )
    if not frames:
        empty = pd.DataFrame()
        return empty, empty, {
            "candidate_signals": 0.0,
            "trades": 0.0,
            "ending_equity": float(initial_capital),
            "total_return": 0.0,
            "cagr": np.nan,
            "max_drawdown": 0.0,
            "sharpe": np.nan,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "payoff_ratio": np.nan,
            "profit_factor": np.nan,
            "exposure": 0.0,
            "avg_positions": 0.0,
            "max_positions_used": 0.0,
            "skipped_slots": 0.0,
            "skipped_no_cash": 0.0,
            "skipped_already_open": 0.0,
        }, pd.DataFrame()

    # Cache exact date -> row index for each instrument.
    date_to_idx: dict[str, dict[pd.Timestamp, int]] = {}
    for stock_id, frame in frames.items():
        date_to_idx[stock_id] = {pd.Timestamp(d): int(i) for i, d in zip(frame.index, frame["date"])}

    if candidates:
        first_date = min(c["entry_date"] for c in candidates)
    else:
        first_date = max(frame["date"].min() for frame in frames.values())
    last_date = max(frame["date"].max() for frame in frames.values())
    calendar = sorted({
        pd.Timestamp(d)
        for frame in frames.values()
        for d in frame.loc[(frame["date"] >= first_date) & (frame["date"] <= last_date), "date"].tolist()
    })

    by_entry_date: dict[pd.Timestamp, list[dict]] = {}
    for c in candidates:
        by_entry_date.setdefault(pd.Timestamp(c["entry_date"]), []).append(c)

    cash = float(initial_capital)
    positions: dict[str, _Position] = {}
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []
    skipped_slots = 0
    skipped_no_cash = 0
    skipped_already_open = 0

    last_equity = float(initial_capital)

    for date in calendar:
        date = pd.Timestamp(date)

        # Entries occur at the open using only cash already available before today's exits.
        todays = by_entry_date.get(date, [])
        todays = sorted(
            todays,
            key=lambda x: (-x["grade_rank"], -x["quality_score"], -x["strength"], -x["volume_ratio"], x["stock_id"]),
        )
        for c in todays:
            sid = c["stock_id"]
            if sid in positions:
                skipped_already_open += 1
                continue
            if len(positions) >= int(max_positions):
                skipped_slots += 1
                continue
            frame = frames.get(sid)
            if frame is None:
                continue
            idx = date_to_idx[sid].get(date)
            if idx is None:
                continue
            entry_price = float(frame.loc[idx, "open"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            target_notional = float(last_equity) * float(position_pct)
            allocation = min(float(cash), target_notional)
            if allocation <= max(float(initial_capital) * 1e-8, 1.0):
                skipped_no_cash += 1
                continue

            shares = allocation / (entry_price * (1.0 + float(one_way_cost)))
            if shares <= 0:
                continue
            cash -= allocation
            stop_price = entry_price * (1.0 - float(stop_loss_pct)) if stop_loss_pct is not None and float(stop_loss_pct) > 0 else None
            take_price = entry_price * (1.0 + float(take_profit_pct)) if take_profit_pct is not None and float(take_profit_pct) > 0 else None
            positions[sid] = _Position(
                stock_id=sid,
                stock_name=c["stock_name"],
                signal_date=pd.Timestamp(c["signal_date"]),
                entry_date=date,
                entry_idx=int(c["entry_idx"]),
                planned_exit_idx=int(c["planned_exit_idx"]),
                shares=float(shares),
                entry_price=entry_price,
                entry_notional=float(allocation),
                stop_price=stop_price,
                take_price=take_price,
                grade=c["grade"],
                quality_score=float(c["quality_score"]),
                signal_strength=float(c["strength"]),
                signal_volume_ratio=float(c["volume_ratio"]),
                signal_momentum20_pct=float(c["momentum20_pct"]) if pd.notna(c["momentum20_pct"]) else np.nan,
                last_mark_price=entry_price,
            )

        # After entries, evaluate today's bar for all open positions.
        exiting: list[str] = []
        for sid, pos in list(positions.items()):
            frame = frames[sid]
            idx = date_to_idx[sid].get(date)
            if idx is None:
                continue
            day = frame.loc[idx]
            day_open = float(day["open"])
            day_high = float(day["high"])
            day_low = float(day["low"])
            day_close = float(day["close"])
            pos.last_mark_price = day_close

            exit_price = None
            exit_reason = None
            if pos.stop_price is not None and day_open <= pos.stop_price:
                exit_price = day_open
                exit_reason = "停損（跳空）"
            elif pos.take_price is not None and day_open >= pos.take_price:
                exit_price = day_open
                exit_reason = "停利（跳空）"
            else:
                hit_stop = pos.stop_price is not None and day_low <= pos.stop_price
                hit_take = pos.take_price is not None and day_high >= pos.take_price
                if hit_stop and hit_take:
                    exit_price = float(pos.stop_price)
                    exit_reason = "同日觸發停損/停利→保守採停損"
                elif hit_stop:
                    exit_price = float(pos.stop_price)
                    exit_reason = "停損"
                elif hit_take:
                    exit_price = float(pos.take_price)
                    exit_reason = "停利"
                elif idx >= pos.planned_exit_idx:
                    exit_price = day_close
                    exit_reason = f"持有{int(hold_days)}日"

            if exit_price is None:
                continue

            proceeds = float(pos.shares) * float(exit_price) * (1.0 - float(one_way_cost))
            cash += proceeds
            gross_return = float(exit_price / pos.entry_price - 1.0)
            net_return = float(proceeds / pos.entry_notional - 1.0)
            pnl = float(proceeds - pos.entry_notional)
            trade_rows.append({
                "股票": sid,
                "名稱": pos.stock_name,
                "訊號日": pos.signal_date,
                "進場日": pos.entry_date,
                "出場日": date,
                "V品質": pos.grade,
                "V品質分數": pos.quality_score,
                "訊號力道": pos.signal_strength,
                "訊號量比": pos.signal_volume_ratio,
                "訊號20日動能%": pos.signal_momentum20_pct,
                "進場價": pos.entry_price,
                "出場價": float(exit_price),
                "投入金額": pos.entry_notional,
                "損益金額": pnl,
                "毛報酬": gross_return,
                "淨報酬": net_return,
                "持有交易日": int(idx - pos.entry_idx + 1),
                "出場原因": exit_reason,
            })
            exiting.append(sid)

        for sid in exiting:
            positions.pop(sid, None)

        invested_value = float(sum(pos.shares * pos.last_mark_price for pos in positions.values()))
        equity_value = float(cash + invested_value)
        equity_rows.append({
            "date": date,
            "equity": equity_value,
            "cash": float(cash),
            "invested_value": invested_value,
            "positions": int(len(positions)),
            "cash_ratio": float(cash / equity_value) if equity_value > 0 else np.nan,
        })
        last_equity = equity_value

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)

    curve_stats = _curve_stats(equity, "equity")
    if trades.empty:
        trade_stats = {
            "win_rate": np.nan,
            "expectancy": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "payoff_ratio": np.nan,
            "profit_factor": np.nan,
        }
    else:
        net = pd.to_numeric(trades["淨報酬"], errors="coerce").dropna()
        wins = net[net > 0]
        losses = net[net < 0]
        avg_win = float(wins.mean()) if len(wins) else np.nan
        avg_loss = float(losses.mean()) if len(losses) else np.nan
        gross_profit = float(pd.to_numeric(trades.loc[trades["損益金額"] > 0, "損益金額"], errors="coerce").sum())
        gross_loss = float(abs(pd.to_numeric(trades.loc[trades["損益金額"] < 0, "損益金額"], errors="coerce").sum()))
        trade_stats = {
            "win_rate": float((net > 0).mean()) if len(net) else np.nan,
            "expectancy": float(net.mean()) if len(net) else np.nan,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": float(avg_win / abs(avg_loss)) if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0 else np.nan,
            "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
        }

    stats = {
        "candidate_signals": float(len(candidates)),
        "trades": float(len(trades)),
        "ending_equity": float(equity["equity"].iloc[-1]) if not equity.empty else float(initial_capital),
        **curve_stats,
        **trade_stats,
        "exposure": float((equity["positions"] > 0).mean()) if not equity.empty else 0.0,
        "avg_positions": float(equity["positions"].mean()) if not equity.empty else 0.0,
        "max_positions_used": float(equity["positions"].max()) if not equity.empty else 0.0,
        "skipped_slots": float(skipped_slots),
        "skipped_no_cash": float(skipped_no_cash),
        "skipped_already_open": float(skipped_already_open),
    }

    if trades.empty:
        contribution = pd.DataFrame(columns=["股票", "名稱", "交易數", "勝率", "平均淨報酬", "損益金額"])
    else:
        rows = []
        for (sid, name), g in trades.groupby(["股票", "名稱"], dropna=False):
            rets = pd.to_numeric(g["淨報酬"], errors="coerce").dropna()
            rows.append({
                "股票": sid,
                "名稱": name,
                "交易數": int(len(g)),
                "勝率": float((rets > 0).mean()) if len(rets) else np.nan,
                "平均淨報酬": float(rets.mean()) if len(rets) else np.nan,
                "損益金額": float(pd.to_numeric(g["損益金額"], errors="coerce").sum()),
            })
        contribution = pd.DataFrame(rows).sort_values("損益金額", ascending=False).reset_index(drop=True)

    return trades, equity, stats, contribution


def build_benchmark_curve(
    benchmark_df: pd.DataFrame,
    portfolio_dates: Iterable[pd.Timestamp],
    initial_capital: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Normalize a buy-and-hold benchmark to the portfolio simulation dates.

    This is a research comparison using daily close prices. It intentionally does not claim
    identical execution timing to the signal portfolio.
    """
    dates = pd.to_datetime(pd.Series(list(portfolio_dates)), errors="coerce").dropna().dt.normalize()
    if dates.empty:
        return pd.DataFrame(), _curve_stats(pd.DataFrame())

    work = benchmark_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    if work.empty:
        return pd.DataFrame(), _curve_stats(pd.DataFrame())

    idx = pd.DataFrame({"date": sorted(dates.unique())})
    merged = pd.merge_asof(idx, work[["date", "close"]], on="date", direction="backward")
    merged = merged.dropna(subset=["close"]).reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame(), _curve_stats(pd.DataFrame())
    first_close = float(merged["close"].iloc[0])
    if first_close <= 0:
        return pd.DataFrame(), _curve_stats(pd.DataFrame())
    merged["equity"] = float(initial_capital) * merged["close"] / first_close
    return merged[["date", "equity", "close"]], _curve_stats(merged, "equity")
