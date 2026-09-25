# 免費台股 AI 多空分析系統 v1

這是一個可以免費部署到 Streamlit Community Cloud 的台股研究工具。第一版提供：

- 台股代號查詢（例如 2330、2317、2454）
- 日 K、MA20、MA60
- RSI、MACD、成交量比
- 自訂 0~100 多空力道分數
- 力道 K 與 V/A 轉折訊號
- 20 日支撐/壓力參考
- Random Forest 預估「未來 5 個交易日收盤高於今天」的機率
- 最後約 25% 歷史資料的留出樣本測試
- 簡單 long/cash 回測
- CSV 下載

> 本專案的力道公式與訊號是自行設計，沒有複製任何第三方網站的專有演算法。

## 最簡單的免費上線方式

### 1. 建立 GitHub 帳號
到 https://github.com/ 註冊免費帳號。

### 2. 建立一個新的 Repository
建議名稱：`tw-stock-ai-v1`

建立時選 **Public** 即可。

### 3. 上傳這個資料夾裡的所有檔案
GitHub Repository 頁面：

1. 按 **Add file**
2. 按 **Upload files**
3. 至少上傳這三個必要檔案：
   - `streamlit_app.py`
   - `stock_engine.py`
   - `requirements.txt`
4. `README.md` 與 `.streamlit/config.toml` 是選配；少了也不影響核心網站執行。
5. 按 **Commit changes**

### 4. 免費部署 Streamlit
到 https://share.streamlit.io/ ，用 GitHub 登入。

依序選：

- Repository：你剛建立的 `tw-stock-ai-v1`
- Branch：`main`
- Main file path：`streamlit_app.py`
- Python：建議 `3.12`

按 **Deploy**。

部署完成後會得到一個 `xxxxx.streamlit.app` 網址。

## 資料來源

程式預設採「自動模式」：

1. 先向 FinMind 取得台股日資料
2. 若失敗或被限流，自動改用 Yahoo Finance / yfinance

FinMind 官方文件目前說明：匿名請求約 300 次/小時；免費註冊並帶 token 可提高到 600 次/小時。第一版不強迫使用 token。

FinMind 文件：
https://finmind.github.io/quickstart/

Streamlit 部署文件：
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 模型怎麼做

模型不是「算命」。它把過去價格資料轉成特徵，例如：

- 1 日 / 5 日報酬
- 價格與 MA20 / MA60 的距離
- RSI(14)
- MACD histogram
- ATR / 波動率
- 成交量相對 20 日均量
- 自訂多空力道分數

目標是分類：

`未來第 5 個交易日的收盤價 > 今天收盤價` → 1，否則 → 0。

使用 Random Forest 輸出機率。

## 回測限制

v1 的回測刻意保持簡單：

- 歷史前約 75% 作訓練資料
- 後約 25% 作測試資料
- 模型機率 >= 55% 時，下一交易日持有股票
- 否則持有現金
- 尚未計入手續費、交易稅、滑價、股利與除權息調整
- 尚不是逐日 walk-forward 重新訓練

因此它只能當「第一版驗證工具」，不能直接視為真實可交易績效。

## 下一版建議

v2 可依序加入：

1. walk-forward 回測
2. 除權息調整價格
3. 多種模型比較
4. 籌碼面與法人買賣超
5. 基本面 / 月營收
6. 自訂股票清單與批次選股
7. 手機版首頁儀表板

## 免責聲明

本工具僅供教育、研究及技術展示，不構成投資建議、招攬或保證。任何投資決策與風險應由使用者自行判斷與承擔。
