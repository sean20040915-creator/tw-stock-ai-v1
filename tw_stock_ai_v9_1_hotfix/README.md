# 台股 AI 多空分析系統 v9.1 — 最新行情 Hotfix

## 這次修正什麼？

原 v9 的個股資料可能停留在前一個交易日，例如 Yahoo Finance 已有 2026-10-01，網站仍顯示 2026-09-30。

主要原因：

1. `load_data()` 使用 Streamlit cache 1 小時，cache key 沒包含台灣日期/刷新時段。
2. Yahoo / FinMind 只要回傳足夠歷史資料就會被接受，沒有檢查最新一根日 K 是否落後官方最新交易日。
3. Streamlit Cloud 的主機時區未必是 Asia/Taipei，原本 `date.today()` 不夠明確。
4. v9 主畫面標題仍誤顯示 v8，這次一併修正。

## v9.1 修正

- 改用 `Asia/Taipei` 計算今天日期。
- 行情 cache 從 1 小時改為 15 分鐘。
- 15 分鐘刷新槽直接加入 cache key；跨日/跨時段會自動重抓。
- 左側新增「🔄 強制更新最新行情」按鈕，可立即清除行情與模型 cache。
- 長期歷史仍由 FinMind / Yahoo 取得。
- 每次取得長期資料後，再用官方最新交易日資料校正：
  - 上市：TWSE `/exchangeReport/STOCK_DAY_ALL`
  - 上櫃：TPEx `/tpex_mainboard_daily_close_quotes`
- 若官方最新交易日比 Yahoo / FinMind 新，會自動補入最後一根 OHLCV。
- v9 主標題誤顯示 v8 的問題一併修正為 v9.1。

## 升級方式

只需要覆蓋 GitHub repository 根目錄的：

- `stock_engine.py`
- `streamlit_app.py`

`portfolio_engine.py` 與 `robustness_engine.py` 沒有必要覆蓋；升級包保留它們只是方便完整檢查。

不要動：

- `data/forward_signals.csv`
- `data/notification_events.csv`
- `forward_watchlist.csv`
- `.github/workflows/*`

Commit 後 Streamlit 會自動重新部署。

## 驗收

用 2330 測試：

- 最新分析日期應由 2026-09-30 更新到 2026-10-01
- 收盤價應為 2510（以 2026-10-01 為例）
- 若 Yahoo 本身仍落後，資料來源文字可能顯示：
  `Yahoo Finance (2330.TW) + TWSE OpenAPI 最新日`
- 左側應看到「🔄 強制更新最新行情」按鈕

