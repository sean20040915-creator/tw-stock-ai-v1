from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from notification_engine import (
    append_events,
    deliver_new_events,
    detect_forward_maturity_events,
    detect_position_events,
    detect_snapshot_events,
    filter_new_events,
    load_positions,
    load_settings,
    read_event_log,
    save_event_log,
)
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
NOTIFICATION_LOG_PATH = ROOT / "data" / "notification_events.csv"
NOTIFICATION_SETTINGS_PATH = ROOT / "notification_settings.csv"
POSITIONS_PATH = ROOT / "paper_positions.csv"


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
    settings = load_settings(NOTIFICATION_SETTINGS_PATH)
    event_log = read_event_log(NOTIFICATION_LOG_PATH)
    positions = load_positions(POSITIONS_PATH)

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
    market_frames: dict[str, pd.DataFrame] = {}
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
            market_frames[ticker] = df

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
            for col in new_df.columns:
                if col not in log.columns:
                    log[col] = pd.NA
            for col in log.columns:
                if col not in new_df.columns:
                    new_df[col] = pd.NA
            log = pd.concat([log, new_df[log.columns]], ignore_index=True)

    if log.empty:
        print(f"No forward records written. ok={ok}, failed={failed}")
        return

    if {"stock_id", "data_date", "model_version"}.issubset(log.columns):
        log = log.drop_duplicates(["stock_id", "data_date", "model_version"], keep="first")
    log = log.sort_values(["data_date", "stock_id", "model_version"]).reset_index(drop=True)

    # Optional paper positions may include a stock not in the forward watchlist.
    # To keep free-API usage bounded, total unique market frames are capped at 25.
    if not positions.empty and len(market_frames) < 25:
        position_ids = []
        for value in positions.get("stock_id", pd.Series(dtype=str)).astype(str).tolist():
            sid = normalize_stock_id(value)
            if sid and sid not in position_ids:
                position_ids.append(sid)
        for ticker in position_ids:
            if ticker in market_frames or len(market_frames) >= 25:
                continue
            try:
                raw, _ = fetch_stock_data(ticker, years=2, source="auto", finmind_token=token)
                market_frames[ticker] = add_indicators(raw)
                print(f"POSITION DATA OK {ticker}")
            except Exception as exc:
                print(f"POSITION DATA ERROR {ticker}: {exc}")

    # v7 notification engine: detect only newly generated events and de-duplicate forever by event_id.
    new_snapshot_df = pd.DataFrame(new_rows)
    candidate_events = []
    candidate_events.extend(detect_snapshot_events(log, new_snapshot_df, settings))
    candidate_events.extend(detect_forward_maturity_events(log, settings))
    candidate_events.extend(detect_position_events(market_frames, positions, settings))
    fresh_events = filter_new_events(candidate_events, event_log)
    if fresh_events:
        fresh_events = deliver_new_events(fresh_events)
        event_log = append_events(event_log, fresh_events)
        print(f"Notification events created: {len(fresh_events)}")
        for event in fresh_events:
            print(
                f"EVENT {event['event_type']} {event['stock_id']} "
                f"telegram={event['telegram_status']} discord={event['discord_status']}"
            )
    else:
        print("No new notification event.")

    log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    save_event_log(event_log, NOTIFICATION_LOG_PATH)
    latest = log["data_date"].astype(str).max() if "data_date" in log.columns else "—"
    print(f"Saved {len(log)} forward rows. latest={latest}, ok={ok}, failed={failed}")
    print(f"Saved {len(event_log)} notification events to {NOTIFICATION_LOG_PATH}")


if __name__ == "__main__":
    main()
