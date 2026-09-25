from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from stock_engine import (
    MODEL_VERSION,
    add_indicators,
    fetch_stock_data,
    forward_snapshot_row,
    normalize_stock_id,
    update_forward_outcomes,
)

ROOT = Path(__file__).resolve().parent
WATCHLIST_PATH = ROOT / "forward_watchlist.csv"
LOG_PATH = ROOT / "data" / "forward_signals.csv"


def read_watchlist() -> pd.DataFrame:
    table = pd.read_csv(WATCHLIST_PATH, dtype={"stock_id": str}).fillna("")
    if "stock_id" not in table.columns:
        raise ValueError("forward_watchlist.csv 必須包含 stock_id 欄位。")
    if "stock_name" not in table.columns:
        table["stock_name"] = ""
    table["stock_id"] = table["stock_id"].map(normalize_stock_id)
    table = table[table["stock_id"].ne("")].drop_duplicates("stock_id", keep="first")
    if len(table) > 25:
        table = table.head(25)
    return table.reset_index(drop=True)


def read_log() -> pd.DataFrame:
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return pd.DataFrame()
    try:
        log = pd.read_csv(LOG_PATH, dtype={"stock_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "stock_id" in log.columns:
        log["stock_id"] = log["stock_id"].map(normalize_stock_id)
    return log


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    watchlist = read_watchlist()
    log = read_log()
    token = os.getenv("FINMIND_TOKEN", "").strip()

    benchmark_df = None
    try:
        raw_benchmark, benchmark_source = fetch_stock_data(
            "0050", years=8, source="auto", finmind_token=token
        )
        benchmark_df = add_indicators(raw_benchmark)
        print(f"Benchmark 0050 OK: {benchmark_source}, {len(benchmark_df)} rows")
    except Exception as exc:
        print(f"Benchmark 0050 failed; continuing without benchmark comparison: {exc}")

    new_rows: list[dict] = []
    ok = 0
    failed = 0

    for item in watchlist.itertuples(index=False):
        ticker = normalize_stock_id(str(item.stock_id))
        name = str(item.stock_name or "")
        try:
            raw, source = fetch_stock_data(
                ticker, years=8, source="auto", finmind_token=token
            )
            df = add_indicators(raw)

            if not log.empty and "stock_id" in log.columns:
                log = update_forward_outcomes(log, ticker, df, benchmark_df=benchmark_df)

            snapshot = forward_snapshot_row(
                ticker,
                df,
                stock_name=name,
                source=source,
                model_version=MODEL_VERSION,
            )
            key_exists = False
            if not log.empty and {"stock_id", "data_date", "model_version"}.issubset(log.columns):
                key_exists = bool(
                    (
                        log["stock_id"].astype(str).eq(ticker)
                        & log["data_date"].astype(str).eq(str(snapshot["data_date"]))
                        & log["model_version"].astype(str).eq(MODEL_VERSION)
                    ).any()
                )
            if not key_exists:
                # Also prevent duplicates among rows created in this same run.
                duplicate_new = any(
                    r["stock_id"] == ticker
                    and r["data_date"] == snapshot["data_date"]
                    and r["model_version"] == MODEL_VERSION
                    for r in new_rows
                )
                if not duplicate_new:
                    new_rows.append(snapshot)
                    print(f"NEW {ticker} {snapshot['data_date']} signal={snapshot['signal'] or '—'}")
            else:
                print(f"EXISTS {ticker} {snapshot['data_date']}")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"ERROR {ticker}: {exc}")

    if ok == 0:
        raise RuntimeError("所有追蹤股票都取得失敗；不寫入前向紀錄，讓 GitHub Actions 顯示失敗以便檢查。")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if log.empty:
            log = new_df
        else:
            # Align old/new schema both directions so upgrades remain append-safe.
            for col in new_df.columns:
                if col not in log.columns:
                    log[col] = pd.NA
            for col in log.columns:
                if col not in new_df.columns:
                    new_df[col] = pd.NA
            log = pd.concat([log, new_df[log.columns]], ignore_index=True)

    if log.empty:
        # Create a header-only CSV from one schema example is impossible without data;
        # an empty file is left untouched and the next successful run will initialize it.
        print(f"No forward records written. ok={ok}, failed={failed}")
        return

    # Final de-duplication never changes the original snapshot fields: first record wins.
    if {"stock_id", "data_date", "model_version"}.issubset(log.columns):
        log = log.drop_duplicates(["stock_id", "data_date", "model_version"], keep="first")
    log = log.sort_values(["data_date", "stock_id", "model_version"]).reset_index(drop=True)
    log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    latest = log["data_date"].astype(str).max() if "data_date" in log.columns else "—"
    print(f"Saved {len(log)} rows to {LOG_PATH}. latest={latest}, ok={ok}, failed={failed}")


if __name__ == "__main__":
    main()
