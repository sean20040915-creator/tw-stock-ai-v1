# 免費台股 AI 多空分析系統 v7

v7 的核心是 **通知中心**。它不更動 v6 的預測演算法，因此模型版本仍維持 `v6-model-1.0`，讓 v6 開始累積的前向驗證可以繼續接在同一模型版本下，不會因為只改通知介面就把成績切斷。

## v7 新功能

### 1. 每日通知事件紀錄

原本的 `Daily forward validation` GitHub Actions 在每天收盤後完成快照與前向驗證更新後，會再檢查通知條件。

事件會寫入：

`data/notification_events.csv`

因此即使完全沒有設定 Telegram 或 Discord，網站的「通知中心」仍然可以看到歷史事件。

### 2. 預設通知條件

`notification_settings.csv` 預設開啟：

- 新 V
- 新 A
- A 級新 V
- 力道快速升溫
  - 3 個交易日力道增加至少 10 分
  - 或 5 個交易日增加至少 15 分
- 研究部位停損 / 停利

「5 日前向結果完成」通知預設關閉，避免每天收到太多成熟樣本訊息；需要時可把 `notify_forward_5d_complete` 改成 `1`。

### 3. 不會每天重複通知同一事件

每個事件都有固定 `event_id`。

同一檔股票、同一天、同一事件類型只會建立一次。GitHub Actions 即使重跑，也不會把相同的新 V / A 一直重複通知。

### 4. 每日摘要推播

v7 不會一個事件發一則訊息，而是把當次新事件整理成一份摘要，再傳到已設定的管道。

目前支援：

- Telegram Bot
- Discord Webhook

兩者都是選用。完全不設定也能使用網站內通知中心。

## Telegram 設定

Telegram 官方 Bot API 使用 BotFather 建立 bot 並取得 token。Token 應視為密碼，不要放進程式碼或公開 CSV。

### A. 建立 Bot

1. 在 Telegram 搜尋官方 `@BotFather`。
2. 輸入 `/newbot`。
3. 依指示設定名稱與 username。
4. BotFather 會提供一組 Bot Token。
5. 打開你剛建立的 bot，按 Start，傳一則訊息給它。

### B. 找到 Chat ID

Bot API 可用 `getUpdates` 取得你剛才傳給 bot 的訊息資訊，其中 `chat.id` 就是 Chat ID。

請注意：查詢網址中會包含 Bot Token，所以不要分享網址或截圖給別人。

### C. 存進 GitHub Secrets

Repository → Settings → Secrets and variables → Actions → New repository secret

建立：

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Discord 設定

Discord 可在伺服器的文字頻道建立 Webhook。

1. Discord Server Settings → Integrations → Webhooks。
2. 建立 Webhook 並選擇接收訊息的頻道。
3. Copy Webhook URL。
4. 到 GitHub Repository → Settings → Secrets and variables → Actions。
5. 新增 Secret：`DISCORD_WEBHOOK_URL`。

Webhook URL 等同通知管道的憑證，不要貼在公開 repository。

## 測試推播

v7 新增第二個 GitHub Actions：

`Test notifications`

設定好 Telegram 或 Discord Secrets 後：

1. GitHub → Actions。
2. 左側選 `Test notifications`。
3. Run workflow。
4. 成功時應收到「台股 AI v7 測試通知成功」。

如果完全沒有設定推播 Secrets，這個測試 workflow 會故意顯示失敗並提示尚未設定通知管道。

## 研究部位停損 / 停利提醒

編輯：

`paper_positions.csv`

格式：

```csv
position_id,stock_id,label,entry_date,entry_price,stop_loss_pct,take_profit_pct,enabled
p001,2330,2330研究部位,2026-09-25,1250,6,12,1
```

欄位：

- `position_id`：自訂、不重複的代號。
- `stock_id`：股票代號。
- `label`：你看得懂的備註。
- `entry_date`：研究進場日期。
- `entry_price`：研究進場價格。
- `stop_loss_pct`：例如 `6` = -6%。
- `take_profit_pct`：例如 `12` = +12%。
- `enabled`：`1` 啟用，`0` 停用。

### 很重要的限制

這不是盤中即時警報。

GitHub Actions 目前是在台股收盤後執行，因此停損 / 停利只是根據該交易日的日 K `open/high/low` 判斷「今天是否曾碰到條件」。

如果同一天的高低價同時跨過停損與停利，日 K 無法知道盤中先後順序，v7 採較保守的「停損先發生」。

如果部位股票不在 `forward_watchlist.csv`，v7 會嘗試額外取得資料；為控制免費 API 使用量，前向追蹤股票加上額外研究部位資料最多處理約 25 個不同代號。最穩定的做法仍是把研究部位股票也放進 `forward_watchlist.csv`。

## 通知中心頁籤

Streamlit v7 多了一個「通知中心」，可以看到：

- 通知事件總數
- 最新事件日
- Telegram / Discord 最近傳送狀態
- 事件類型篩選
- 新 V / A
- A 級 V
- 力道快速升溫
- 研究部位停損 / 停利
- 通知規則
- `paper_positions.csv` 目前內容
- 通知事件 CSV 下載

## v6 → v7 升級時最重要的資料保護

**不要覆蓋你 GitHub 裡既有的 `data/forward_signals.csv`。**

它是從 v6 開始累積的真實前向紀錄。v7 沒有改模型，因此應直接延續。

如果你已自訂 `forward_watchlist.csv`，也保留原本檔案即可；v7 的格式沒有改。

建議直接使用本次提供的 **v7_upgrade** 更新包，它刻意不包含上述兩個既有資料檔。

## v7 需要新增 / 更新的檔案

更新：

- `streamlit_app.py`
- `forward_tracker.py`
- `.github/workflows/forward_tracker.yml`
- `README.md`
- `升級步驟.txt`

新增：

- `notification_engine.py`
- `test_notification.py`
- `notification_settings.csv`
- `paper_positions.csv`
- `data/notification_events.csv`
- `.github/workflows/test_notification.yml`

`stock_engine.py` 與 requirements 在 v7 沒有新增第三方套件需求，但完整範本仍會一起保留。

## GitHub Secrets 安全

通知憑證應只放在 GitHub Actions Secrets。v7 不會把 Telegram Bot Token、Telegram Chat ID、Discord Webhook URL 或 FinMind Token 寫入通知事件 CSV。

GitHub Actions log 也不會主動輸出這些 secret 值。

## v7 測試順序

1. 上傳 v7 更新檔並 Commit。
2. Streamlit 顯示「免費台股 AI 多空分析系統 v7」。
3. 確認「前向驗證」仍看得到 v6 累積資料。
4. 確認新「通知中心」可以打開。
5. GitHub Actions 中看到：
   - `Daily forward validation`
   - `Test notifications`
6. 先手動跑一次 `Daily forward validation`。
7. 若當天有符合條件事件，`data/notification_events.csv` 應出現資料。
8. 若想用推播，再設定 Telegram / Discord Secret。
9. 跑 `Test notifications` 確認推播。

## 重要聲明

v7 是教育與研究工具，不是投資建議，也不是即時交易或自動下單系統。

通知只代表程式規則被觸發，不代表適合買進、賣出或持有。免費行情與 API 也可能有延遲、缺漏或調整。
