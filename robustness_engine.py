from __future__ import annotations

from itertools import product
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from portfolio_engine import portfolio_backtest
from stock_engine import add_indicators, v_signal_quality


PARAM_COLUMNS = ["min_strength", "min_volume_ratio", "hold_days", "stop_loss_pct", "take_profit_pct"]
DISPLAY_NAMES = {
    "min_strength": "最低力道",
    "min_volume_ratio": "最低量比",
    "hold_days": "持有日",
    "stop_loss_pct": "停損%",
    "take_profit_pct": "停利%",
}



def precompute_v_quality(data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Pre-compute V quality once so grid scans do not repeat the expensive history check."""
    prepared: Dict[str, pd.DataFrame] = {}
    for stock_id, raw in data_map.items():
        work = raw.copy()
        if "signal" not in work.columns or "strength_score" not in work.columns:
            work = add_indicators(work)
        work["_v_quality_grade"] = pd.Series([None] * len(work), index=work.index, dtype="object")
        work["_v_quality_score"] = np.nan
        v_indices = work.index[work["signal"].eq("V")].tolist() if "signal" in work.columns else []
        for idx in v_indices:
            try:
                quality = v_signal_quality(work, signal_index=int(idx), horizon=5)
                grade = str(quality.get("等級", "—"))
                score = quality.get("分數")
                work.loc[idx, "_v_quality_grade"] = grade
                work.loc[idx, "_v_quality_score"] = float(score) if pd.notna(score) else 0.0
            except Exception:
                work.loc[idx, "_v_quality_grade"] = "D"
                work.loc[idx, "_v_quality_score"] = 0.0
        prepared[str(stock_id)] = work
    return prepared

def _clean_values(values: Iterable[float | int], cast=float) -> list:
    out = []
    for value in values:
        try:
            v = cast(value)
        except (TypeError, ValueError):
            continue
        if v not in out:
            out.append(v)
    return sorted(out)


def combination_count(
    strength_values: Iterable[float],
    volume_values: Iterable[float],
    hold_values: Iterable[int],
    stop_values: Iterable[float],
    take_values: Iterable[float],
) -> int:
    lengths = [
        len(_clean_values(strength_values, float)),
        len(_clean_values(volume_values, float)),
        len(_clean_values(hold_values, int)),
        len(_clean_values(stop_values, float)),
        len(_clean_values(take_values, float)),
    ]
    if any(n == 0 for n in lengths):
        return 0
    result = 1
    for n in lengths:
        result *= n
    return int(result)


def _parameter_grid(
    strength_values: Iterable[float],
    volume_values: Iterable[float],
    hold_values: Iterable[int],
    stop_values: Iterable[float],
    take_values: Iterable[float],
) -> list[dict]:
    strengths = _clean_values(strength_values, float)
    volumes = _clean_values(volume_values, float)
    holds = _clean_values(hold_values, int)
    stops = _clean_values(stop_values, float)
    takes = _clean_values(take_values, float)
    return [
        {
            "min_strength": float(s),
            "min_volume_ratio": float(v),
            "hold_days": int(h),
            "stop_loss_pct": float(sl),
            "take_profit_pct": float(tp),
        }
        for s, v, h, sl, tp in product(strengths, volumes, holds, stops, takes)
    ]


def _safe_float(value, default=np.nan) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def _add_neighborhood_metrics(results: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if results.empty:
        return results

    out = results.copy().reset_index(drop=True)
    value_maps: dict[str, dict[float, int]] = {}
    for col in PARAM_COLUMNS:
        vals = sorted(pd.to_numeric(out[col], errors="coerce").dropna().unique().tolist())
        value_maps[col] = {float(v): i for i, v in enumerate(vals)}

    coords = []
    for _, row in out.iterrows():
        coord = tuple(value_maps[col][float(row[col])] for col in PARAM_COLUMNS)
        coords.append(coord)
    coord_to_idx = {coord: i for i, coord in enumerate(coords)}

    neighborhood_rows: list[dict] = []
    for i, coord in enumerate(coords):
        neighbor_indices = {i}
        for dim, col in enumerate(PARAM_COLUMNS):
            if len(value_maps[col]) <= 1:
                continue
            for delta in (-1, 1):
                candidate = list(coord)
                candidate[dim] += delta
                candidate = tuple(candidate)
                if candidate in coord_to_idx:
                    neighbor_indices.add(coord_to_idx[candidate])

        nbh = out.loc[sorted(neighbor_indices)].copy()
        eligible = nbh[pd.to_numeric(nbh["trades"], errors="coerce") >= int(min_trades)].copy()
        base = eligible if not eligible.empty else nbh

        median_cagr = pd.to_numeric(base["cagr"], errors="coerce").median()
        median_return = pd.to_numeric(base["total_return"], errors="coerce").median()
        median_expectancy = pd.to_numeric(base["expectancy"], errors="coerce").median()
        median_mdd = pd.to_numeric(base["max_drawdown"], errors="coerce").median()
        median_sharpe = pd.to_numeric(base["sharpe"], errors="coerce").median()
        pos_exp_ratio = float((pd.to_numeric(base["expectancy"], errors="coerce") > 0).mean()) if len(base) else np.nan
        pos_return_ratio = float((pd.to_numeric(base["total_return"], errors="coerce") > 0).mean()) if len(base) else np.nan
        median_trades = pd.to_numeric(base["trades"], errors="coerce").median()

        # Heuristic score for locating plateaus, not for predicting future returns.
        sample_score = np.clip(_safe_float(median_trades, 0.0) / max(int(min_trades) * 2, 1), 0.0, 1.0)
        pos_score = np.nanmean([pos_exp_ratio, pos_return_ratio])
        if not np.isfinite(pos_score):
            pos_score = 0.0
        exp_score = np.clip((_safe_float(median_expectancy, -0.01) + 0.01) / 0.03, 0.0, 1.0)
        sharpe_score = np.clip((_safe_float(median_sharpe, -0.5) + 0.5) / 2.0, 0.0, 1.0)
        dd_score = np.clip(1.0 - abs(_safe_float(median_mdd, -1.0)) / 0.40, 0.0, 1.0)
        score = 100.0 * (0.35 * pos_score + 0.20 * exp_score + 0.20 * sharpe_score + 0.15 * dd_score + 0.10 * sample_score)

        robust = bool(
            len(base) >= 3
            and _safe_float(median_trades, 0) >= int(min_trades)
            and _safe_float(pos_exp_ratio, 0) >= 0.65
            and _safe_float(pos_return_ratio, 0) >= 0.65
            and _safe_float(median_expectancy, -1) > 0
            and _safe_float(median_cagr, -1) > 0
        )

        neighborhood_rows.append({
            "neighbor_count": int(len(nbh)),
            "eligible_neighbor_count": int(len(eligible)),
            "neighbor_median_trades": _safe_float(median_trades),
            "neighbor_median_total_return": _safe_float(median_return),
            "neighbor_median_cagr": _safe_float(median_cagr),
            "neighbor_median_expectancy": _safe_float(median_expectancy),
            "neighbor_median_max_drawdown": _safe_float(median_mdd),
            "neighbor_median_sharpe": _safe_float(median_sharpe),
            "neighbor_positive_expectancy_ratio": _safe_float(pos_exp_ratio),
            "neighbor_positive_return_ratio": _safe_float(pos_return_ratio),
            "robustness_score": float(score),
            "robust_region": robust,
        })

    metrics = pd.DataFrame(neighborhood_rows)
    out = pd.concat([out, metrics], axis=1)

    # Isolated peak: strong own result but nearby settings do not confirm it.
    cagr_q75 = pd.to_numeric(out["cagr"], errors="coerce").quantile(0.75)
    out["isolated_peak"] = (
        (pd.to_numeric(out["cagr"], errors="coerce") >= cagr_q75)
        & (pd.to_numeric(out["neighbor_positive_expectancy_ratio"], errors="coerce") < 0.60)
    )
    return out


def run_parameter_scan(
    data_map: Dict[str, pd.DataFrame],
    stock_names: dict[str, str] | None,
    initial_capital: float,
    allowed_grades: tuple[str, ...],
    strength_values: Iterable[float],
    volume_values: Iterable[float],
    hold_values: Iterable[int],
    stop_values: Iterable[float],
    take_values: Iterable[float],
    min_momentum_20: float | None,
    one_way_cost: float,
    max_positions: int,
    position_pct: float,
    min_trades: int = 8,
    max_combinations: int = 120,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict]:
    """Run a bounded Cartesian parameter scan using the same v8 portfolio simulator.

    The function does not pick a single 'best' strategy. It adds local-neighborhood metrics
    so the UI can distinguish broad plateaus from isolated historical peaks.
    """
    grid = _parameter_grid(strength_values, volume_values, hold_values, stop_values, take_values)
    if not grid:
        raise ValueError("參數範圍不可為空。")
    if len(grid) > int(max_combinations):
        raise ValueError(f"參數組合共 {len(grid)} 組，超過上限 {int(max_combinations)} 組。請縮小掃描範圍。")

    prepared_map = precompute_v_quality(data_map)

    rows = []
    for i, params in enumerate(grid, start=1):
        trades, equity, stats, _ = portfolio_backtest(
            prepared_map,
            stock_names=stock_names or {},
            initial_capital=float(initial_capital),
            allowed_grades=tuple(allowed_grades),
            min_strength=float(params["min_strength"]),
            min_volume_ratio=float(params["min_volume_ratio"]),
            min_momentum_20=min_momentum_20,
            hold_days=int(params["hold_days"]),
            stop_loss_pct=float(params["stop_loss_pct"]) / 100.0 if float(params["stop_loss_pct"]) > 0 else None,
            take_profit_pct=float(params["take_profit_pct"]) / 100.0 if float(params["take_profit_pct"]) > 0 else None,
            one_way_cost=float(one_way_cost),
            max_positions=int(max_positions),
            position_pct=float(position_pct),
        )
        row = {
            **params,
            "candidate_signals": _safe_float(stats.get("candidate_signals"), 0),
            "trades": _safe_float(stats.get("trades"), 0),
            "ending_equity": _safe_float(stats.get("ending_equity")),
            "total_return": _safe_float(stats.get("total_return")),
            "cagr": _safe_float(stats.get("cagr")),
            "max_drawdown": _safe_float(stats.get("max_drawdown")),
            "sharpe": _safe_float(stats.get("sharpe")),
            "win_rate": _safe_float(stats.get("win_rate")),
            "expectancy": _safe_float(stats.get("expectancy")),
            "payoff_ratio": _safe_float(stats.get("payoff_ratio")),
            "profit_factor": _safe_float(stats.get("profit_factor")),
            "exposure": _safe_float(stats.get("exposure")),
            "max_positions_used": _safe_float(stats.get("max_positions_used"), 0),
            "skipped_slots": _safe_float(stats.get("skipped_slots"), 0),
        }
        rows.append(row)
        if callable(progress_callback):
            progress_callback(i, len(grid), row)

    results = pd.DataFrame(rows)
    results = _add_neighborhood_metrics(results, int(min_trades))

    eligible = results[pd.to_numeric(results["trades"], errors="coerce") >= int(min_trades)].copy()
    robust = eligible[eligible["robust_region"].eq(True)].copy() if not eligible.empty else eligible
    meta = {
        "combinations": int(len(results)),
        "eligible_combinations": int(len(eligible)),
        "robust_combinations": int(len(robust)),
        "min_trades": int(min_trades),
        "median_cagr": _safe_float(pd.to_numeric(eligible["cagr"], errors="coerce").median()) if not eligible.empty else np.nan,
        "median_expectancy": _safe_float(pd.to_numeric(eligible["expectancy"], errors="coerce").median()) if not eligible.empty else np.nan,
        "positive_expectancy_ratio": float((pd.to_numeric(eligible["expectancy"], errors="coerce") > 0).mean()) if not eligible.empty else np.nan,
        "positive_return_ratio": float((pd.to_numeric(eligible["total_return"], errors="coerce") > 0).mean()) if not eligible.empty else np.nan,
        "median_max_drawdown": _safe_float(pd.to_numeric(eligible["max_drawdown"], errors="coerce").median()) if not eligible.empty else np.nan,
    }
    return results, meta


def parameter_sensitivity_summary(results: pd.DataFrame, min_trades: int = 8) -> dict[str, pd.DataFrame]:
    """Summarize how results change when each individual parameter changes."""
    output: dict[str, pd.DataFrame] = {}
    if results.empty:
        return output
    eligible = results[pd.to_numeric(results["trades"], errors="coerce") >= int(min_trades)].copy()
    if eligible.empty:
        return output

    for col in PARAM_COLUMNS:
        rows = []
        for value, g in eligible.groupby(col, dropna=False):
            rows.append({
                DISPLAY_NAMES[col]: value,
                "組合數": int(len(g)),
                "交易數中位數": float(pd.to_numeric(g["trades"], errors="coerce").median()),
                "累積報酬中位數": float(pd.to_numeric(g["total_return"], errors="coerce").median()),
                "年化報酬中位數": float(pd.to_numeric(g["cagr"], errors="coerce").median()),
                "期望報酬中位數": float(pd.to_numeric(g["expectancy"], errors="coerce").median()),
                "最大回撤中位數": float(pd.to_numeric(g["max_drawdown"], errors="coerce").median()),
                "Sharpe中位數": float(pd.to_numeric(g["sharpe"], errors="coerce").median()),
                "正期望比例": float((pd.to_numeric(g["expectancy"], errors="coerce") > 0).mean()),
                "正報酬比例": float((pd.to_numeric(g["total_return"], errors="coerce") > 0).mean()),
                "穩健區比例": float(g["robust_region"].astype(bool).mean()),
            })
        output[col] = pd.DataFrame(rows).sort_values(DISPLAY_NAMES[col]).reset_index(drop=True)
    return output


def heatmap_slice(
    results: pd.DataFrame,
    hold_days: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    metric: str = "neighbor_median_cagr",
) -> pd.DataFrame:
    """Return a strength x volume pivot for one exit-rule slice."""
    if results.empty or metric not in results.columns:
        return pd.DataFrame()
    q = results[
        (pd.to_numeric(results["hold_days"], errors="coerce") == int(hold_days))
        & np.isclose(pd.to_numeric(results["stop_loss_pct"], errors="coerce"), float(stop_loss_pct))
        & np.isclose(pd.to_numeric(results["take_profit_pct"], errors="coerce"), float(take_profit_pct))
    ].copy()
    if q.empty:
        return pd.DataFrame()
    return q.pivot_table(
        index="min_volume_ratio",
        columns="min_strength",
        values=metric,
        aggfunc="median",
    ).sort_index(ascending=False)
