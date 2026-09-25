from __future__ import annotations

import os
import sys

from notification_engine import send_discord, send_telegram


def main() -> None:
    message = "✅ 台股 AI v7 測試通知成功。之後 Daily forward validation 有新事件時，會由 GitHub Actions 自動推播。"
    configured = 0
    failed = []

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if tg_token or tg_chat:
        configured += 1
        ok, status = send_telegram(message, tg_token, tg_chat)
        print(f"Telegram: {status}")
        if not ok:
            failed.append(f"Telegram={status}")

    discord_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if discord_url:
        configured += 1
        ok, status = send_discord(message, discord_url)
        print(f"Discord: {status}")
        if not ok:
            failed.append(f"Discord={status}")

    if configured == 0:
        raise RuntimeError(
            "尚未設定通知管道。請在 Repository → Settings → Secrets and variables → Actions "
            "加入 Telegram 或 Discord 的 secrets 後再測試。"
        )
    if failed:
        raise RuntimeError("；".join(failed))
    print("At least one notification channel succeeded.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
