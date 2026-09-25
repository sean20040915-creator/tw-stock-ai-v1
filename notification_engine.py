from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

EVENT_COLUMNS = [
    "event_id",
    "created_at",
    "data_date",
    "stock_id",
    "stock_name",
    "event_type",
    "title",
    "message",
    "telegram_status",
    "discord_status",
]

DEFAULT_SETTINGS = {
    "notify_new_v": True,
    "notify_new_a": True,
    "notify_a_grade_v": True,
    "notify_rapid_warming": True,
    "rapid_3d_threshold": 10.0,
    "rapid_5d_threshold": 15.0,
    "notify_forward_5d_complete": False,
    "notify_stop_take": True,
}


def _bool_value(value, default: bool) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "是", "啟用"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", "停用"}:
        return False
    return default


def load_settings(path: str | Path) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return settings
    try:
        table = pd.read_csv(path)
    except Exception:
        return settings
    if table.empty:
        return settings
    row = table.iloc[0].to_dict()
    for key, default in DEFAULT_SETTINGS.items():
        if key not in row:
            continue
        if isinstance(default, bool):
            settings[key] = _bool_value(row.get(key), default)
        else:
            try:
                settings[key] = float(row.get(key))
            except Exception:
                settings[key] = default
    return settings


def read_event_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    try:
        table = pd.read_csv(path, dtype={"stock_id": str, "event_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    for col in EVENT_COLUMNS:
        if col not in table.columns:
            table[col] = ""
    return table[EVENT_COLUMNS]


def save_event_log(table: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    work = table.copy()
    for col in EVENT_COLUMNS:
        if col not in work.columns:
            work[col] = ""
    work = work[EVENT_COLUMNS]
    work = work.drop_duplicates("event_id", keep="first")
    if "data_date" in work.columns:
        work = work.sort_values(["data_date", "created_at", "stock_id"], na_position="last")
    work.to_csv(path, index=False, encoding="utf-8-sig")


def _event_id(*parts: object) -> str:
    raw = "|".join(str(x) for x in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _event(
    data_date: str,
    stock_id: str,
    stock_name: str,
    event_type: str,
    title: str,
    message: str,
    unique_suffix: str = "",
) -> dict:
    created_at = pd.Timestamp.now(tz="Asia/Taipei").isoformat()
    return {
        "event_id": _event_id(data_date, stock_id, event_type, unique_suffix),
        "created_at": created_at,
        "data_date": data_date,
        "stock_id": str(stock_id),
        "stock_name": str(stock_name or ""),
        "event_type": event_type,
        "title": title,
        "message": message,
        "telegram_status": "pending",
        "discord_status": "pending",
    }


def detect_snapshot_events(
    full_log: pd.DataFrame,
    new_snapshots: pd.DataFrame,
    settings: dict,
) -> list[dict]:
    events: list[dict] = []
    if new_snapshots is None or new_snapshots.empty:
        return events

    log = full_log.copy()
    log["data_date"] = pd.to_datetime(log["data_date"], errors="coerce")
    log["strength_score"] = pd.to_numeric(log.get("strength_score"), errors="coerce")

    for _, row in new_snapshots.iterrows():
        sid = str(row.get("stock_id", ""))
        name = str(row.get("stock_name", "") or "")
        data_date = str(row.get("data_date", ""))
        signal = str(row.get("signal", "") or "")
        grade = str(row.get("v_grade", "—") or "—")
        strength = pd.to_numeric(pd.Series([row.get("strength_score")]), errors="coerce").iloc[0]
        ai_prob = pd.to_numeric(pd.Series([row.get("ai_up_5d_prob")]), errors="coerce").iloc[0]

        label = f"{sid} {name}".strip()
        strength_text = "—" if pd.isna(strength) else f"{float(strength):.0f}"
        prob_text = "—" if pd.isna(ai_prob) else f"{float(ai_prob) * 100:.1f}%"

        if signal == "V":
            if grade == "A" and settings.get("notify_a_grade_v", True):
                events.append(_event(
                    data_date, sid, name, "A_GRADE_V", "A 級新 V",
                    f"{label} 出現 A 級 V｜力道 {strength_text}｜AI 5日上漲機率 {prob_text}",
                ))
            elif settings.get("notify_new_v", True):
                events.append(_event(
                    data_date, sid, name, "NEW_V", "新 V 訊號",
                    f"{label} 出現 V 訊號｜品質 {grade}｜力道 {strength_text}｜AI 5日上漲機率 {prob_text}",
                ))
        elif signal == "A" and settings.get("notify_new_a", True):
            events.append(_event(
                data_date, sid, name, "NEW_A", "新 A 訊號",
                f"{label} 出現 A 訊號｜力道 {strength_text}｜AI 5日上漲機率 {prob_text}",
            ))

        if settings.get("notify_rapid_warming", True):
            hist = log[
                log["stock_id"].astype(str).eq(sid)
                & log["model_version"].astype(str).eq(str(row.get("model_version", "")))
                & log["data_date"].notna()
            ].sort_values("data_date").drop_duplicates("data_date", keep="first")
            # Current snapshot should be included in full_log. Compare by trading-day positions.
            current_pos_arr = np.flatnonzero(hist["data_date"].dt.date.astype(str).eq(data_date).to_numpy())
            if len(current_pos_arr):
                pos = int(current_pos_arr[-1])
                delta3 = np.nan
                delta5 = np.nan
                if pos >= 3:
                    delta3 = float(hist.iloc[pos]["strength_score"] - hist.iloc[pos - 3]["strength_score"])
                if pos >= 5:
                    delta5 = float(hist.iloc[pos]["strength_score"] - hist.iloc[pos - 5]["strength_score"])
                hit3 = pd.notna(delta3) and delta3 >= float(settings.get("rapid_3d_threshold", 10.0))
                hit5 = pd.notna(delta5) and delta5 >= float(settings.get("rapid_5d_threshold", 15.0))
                if hit3 or hit5:
                    pieces = []
                    if pd.notna(delta3):
                        pieces.append(f"3日 {delta3:+.1f}")
                    if pd.notna(delta5):
                        pieces.append(f"5日 {delta5:+.1f}")
                    events.append(_event(
                        data_date, sid, name, "RAPID_WARMING", "力道快速升溫",
                        f"{label} 力道快速升溫｜{' / '.join(pieces)}｜目前力道 {strength_text}",
                    ))
    return events


def detect_forward_maturity_events(full_log: pd.DataFrame, settings: dict) -> list[dict]:
    if not settings.get("notify_forward_5d_complete", False) or full_log is None or full_log.empty:
        return []
    work = full_log.copy()
    if "evaluated_5d_at" not in work.columns:
        return []
    today_date = pd.Timestamp.now(tz="Asia/Taipei").date().isoformat()
    completed = work[work["evaluated_5d_at"].astype(str).eq(today_date)].copy()
    events: list[dict] = []
    for _, row in completed.iterrows():
        ret = pd.to_numeric(pd.Series([row.get("ret_5d")]), errors="coerce").iloc[0]
        if pd.isna(ret):
            continue
        sid = str(row.get("stock_id", ""))
        name = str(row.get("stock_name", "") or "")
        signal = str(row.get("signal", "") or "—")
        base_date = str(row.get("data_date", ""))
        label = f"{sid} {name}".strip()
        events.append(_event(
            today_date,
            sid,
            name,
            "FORWARD_5D_COMPLETE",
            "5 日前向結果完成",
            f"{label}｜{base_date} 的 {signal} / 每日快照已完成 5 日驗證：{float(ret) * 100:+.2f}%",
            unique_suffix=base_date,
        ))
    return events


def load_positions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    columns = ["position_id", "stock_id", "label", "entry_date", "entry_price", "stop_loss_pct", "take_profit_pct", "enabled"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        table = pd.read_csv(path, dtype={"position_id": str, "stock_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in table.columns:
            table[col] = ""
    return table[columns]


def detect_position_events(
    market_frames: dict[str, pd.DataFrame],
    positions: pd.DataFrame,
    settings: dict,
) -> list[dict]:
    if not settings.get("notify_stop_take", True) or positions is None or positions.empty:
        return []
    events: list[dict] = []
    for _, row in positions.iterrows():
        if not _bool_value(row.get("enabled"), True):
            continue
        sid = str(row.get("stock_id", "")).strip()
        if sid.endswith(".0"):
            sid = sid[:-2]
        position_id = str(row.get("position_id", "")).strip() or sid
        label_text = str(row.get("label", "") or "")
        frame = market_frames.get(sid)
        if frame is None or frame.empty:
            continue
        work = frame.sort_values("date").reset_index(drop=True)
        latest = work.iloc[-1]
        latest_date = pd.Timestamp(latest["date"]).date().isoformat()
        try:
            entry_date = pd.Timestamp(row.get("entry_date")).date()
            entry_price = float(row.get("entry_price"))
            stop_pct = float(row.get("stop_loss_pct")) / 100.0
            take_pct = float(row.get("take_profit_pct")) / 100.0
        except Exception:
            continue
        if not np.isfinite(entry_price) or entry_price <= 0 or pd.Timestamp(latest["date"]).date() <= entry_date:
            continue
        stop_price = entry_price * (1 - stop_pct) if stop_pct > 0 else None
        take_price = entry_price * (1 + take_pct) if take_pct > 0 else None
        day_open = float(latest["open"])
        day_high = float(latest["high"])
        day_low = float(latest["low"])
        event_type = None
        hit_price = None
        reason = None
        if stop_price is not None and day_open <= stop_price:
            event_type, hit_price, reason = "POSITION_STOP", day_open, "跳空跌破停損"
        elif take_price is not None and day_open >= take_price:
            event_type, hit_price, reason = "POSITION_TAKE", day_open, "跳空突破停利"
        else:
            hit_stop = stop_price is not None and day_low <= stop_price
            hit_take = take_price is not None and day_high >= take_price
            if hit_stop and hit_take:
                event_type, hit_price, reason = "POSITION_STOP", float(stop_price), "同日碰到停損與停利，保守採停損"
            elif hit_stop:
                event_type, hit_price, reason = "POSITION_STOP", float(stop_price), "觸及停損"
            elif hit_take:
                event_type, hit_price, reason = "POSITION_TAKE", float(take_price), "觸及停利"
        if event_type:
            display = f"{sid} {label_text}".strip()
            pnl = hit_price / entry_price - 1
            title = "研究部位停損提醒" if event_type == "POSITION_STOP" else "研究部位停利提醒"
            position_event = _event(
                latest_date,
                sid,
                label_text,
                event_type,
                title,
                f"{display}｜{reason}｜進場 {entry_price:.2f} → 觸發價 {hit_price:.2f}（{pnl * 100:+.2f}%）",
                unique_suffix=position_id,
            )
            # A position alert should fire only once for that position_id + trigger type,
            # even if price remains beyond the level on later days. Reuse a new position_id
            # when starting a new research trade in the same ticker.
            position_event["event_id"] = _event_id(position_id, event_type)
            events.append(position_event)
    return events


def filter_new_events(events: Iterable[dict], existing_log: pd.DataFrame) -> list[dict]:
    existing = set(existing_log.get("event_id", pd.Series(dtype=str)).astype(str).tolist()) if existing_log is not None else set()
    out = []
    seen = set()
    for event in events:
        eid = str(event.get("event_id", ""))
        if not eid or eid in existing or eid in seen:
            continue
        seen.add(eid)
        out.append(event)
    return out


def build_digest(events: list[dict], max_chars: int = 1800) -> str:
    if not events:
        return ""
    data_date = max(str(e.get("data_date", "")) for e in events)
    lines = [f"📈 台股 AI v7 通知｜{data_date}"]
    for event in events:
        icon = {
            "A_GRADE_V": "🟢",
            "NEW_V": "🟢",
            "NEW_A": "🔴",
            "RAPID_WARMING": "🔥",
            "FORWARD_5D_COMPLETE": "🧪",
            "POSITION_STOP": "🛑",
            "POSITION_TAKE": "🎯",
        }.get(str(event.get("event_type")), "•")
        line = f"{icon} {event.get('message', '')}"
        candidate = "\n".join(lines + [line])
        if len(candidate) > max_chars:
            lines.append("…其餘事件請到網站『通知中心』查看。")
            break
        lines.append(line)
    lines.append("僅供研究，不是投資建議。")
    return "\n".join(lines)


def send_telegram(message: str, token: str, chat_id: str) -> tuple[bool, str]:
    token = (token or "").strip()
    chat_id = (chat_id or "").strip()
    if not token or not chat_id:
        return False, "not_configured"
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
            timeout=20,
        )
        if 200 <= response.status_code < 300:
            return True, "sent"
        return False, f"failed_http_{response.status_code}"
    except Exception as exc:
        return False, f"failed_{type(exc).__name__}"


def send_discord(message: str, webhook_url: str) -> tuple[bool, str]:
    webhook_url = (webhook_url or "").strip()
    if not webhook_url:
        return False, "not_configured"
    try:
        response = requests.post(webhook_url, json={"content": message}, timeout=20)
        if response.status_code in (200, 204):
            return True, "sent"
        return False, f"failed_http_{response.status_code}"
    except Exception as exc:
        return False, f"failed_{type(exc).__name__}"


def deliver_new_events(events: list[dict]) -> list[dict]:
    if not events:
        return events
    message = build_digest(events)
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
    _, tg_status = send_telegram(message, telegram_token, telegram_chat_id)
    _, dc_status = send_discord(message, discord_webhook)
    for event in events:
        event["telegram_status"] = tg_status
        event["discord_status"] = dc_status
    return events


def append_events(existing_log: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    if not events:
        return existing_log.copy()
    new = pd.DataFrame(events)
    base = existing_log.copy()
    for col in EVENT_COLUMNS:
        if col not in base.columns:
            base[col] = ""
        if col not in new.columns:
            new[col] = ""
    out = pd.concat([base[EVENT_COLUMNS], new[EVENT_COLUMNS]], ignore_index=True)
    return out.drop_duplicates("event_id", keep="first")
