# 免費台股 AI 多空分析系統 v6

v6 的核心不是再加更多技術指標，而是加入 **前向驗證（forward validation / paper tracking）**：從 v6 上線當天開始，把當天真正看到的訊號、力道與 AI 機率留下來，等未來 5 / 10 / 20 個交易日真的發生後，再補上結果。

這讓「歷史回測」與「上線後真實紀錄」完全分開。

## v6 新功能

### 1. 每日不可回頭改答案的快照

GitHub Actions 每週一到週五、台北時間 17:30 自動執行：

1. 讀取 `forward_watchlist.csv`
2. 取得每檔最新行情
3. 計算四色力道、V/A、V 品質與 AI 5 日上漲機率
4. 把最新交易日的快照寫入 `data/forward_signals.csv`
5. 若 5 / 10 / 20 個交易日已到期，才補上實際結果
6. 自動 Commit 回 GitHub

同一檔股票、同一交易日、同一模型版本不會重複新增。

### 2. 模型版本鎖定

目前版本：

`v6-model-1.0`

之後若更改模型或訊號邏輯，應改用新版本名稱，避免把不同演算法的成績混在一起。

### 3. 5 / 10 / 20 日真實前向結果

每筆每日快照會逐步補上：

- 5 日後報酬
- 10 日後報酬
- 20 日後報酬
- MFE：期間內最大有利漲幅
- MAE：期間內最大不利跌幅
- 同期間 0050 報酬
- 相對 0050 超額報酬

未來結果欄位只在到期後填入；已填入的結果不會在之後重新覆寫。

### 4. V / A Forward Track Record

新的「前向驗證」頁籤會顯示：

- 真正上線後的 V / A 樣本數
- 方向勝率
- 平均方向報酬
- 中位方向報酬
- 相對 0050 超額表現
- 平均 MFE / MAE

剛上線時樣本會是 0，這是正常的。第一批 5 日成績需要等 5 個交易日後才會出現。

### 5. AI 機率校準

每天所有追蹤股票都會保留 AI 5 日上漲機率，不只 V/A 訊號。

等 5 日結果累積後，系統會比較：

- AI 說 `<40%`
- `40–49%`
- `50–59%`
- `60–69%`
- `>=70%`

各區間後來實際上漲的比例。

這比單看模型 accuracy 更容易看出「67% 到底是不是真的接近 67%」。

### 6. 資料健康檢查

「前向驗證」頁面會顯示：

- 目前個股最新行情日期
- 最新前向快照日期
- 模型版本
- 已完成 5 日驗證的紀錄數
- 目前個股最新交易日是否已存在前向快照

## 自動追蹤股票池

預設在 `forward_watchlist.csv`：

- 2330 台積電
- 2317 鴻海
- 2454 聯發科
- 2308 台達電
- 2382 廣達
- 2881 富邦金
- 2882 國泰金
- 2891 中信金
- 2303 聯電
- 0050 元大台灣50

你可以直接在 GitHub 打開 `forward_watchlist.csv`，按鉛筆編輯。建議先維持 10 檔；v6 最多讀前 25 檔，以控制免費 API 與 GitHub Actions 執行量。

CSV 格式：

```csv
stock_id,stock_name
2330,台積電
2317,鴻海
```

## v5 → v6 升級需要上傳的檔案

這一次不只 3 個檔案，請把以下內容全部放進原本 repository：

- `streamlit_app.py`
- `stock_engine.py`
- `requirements.txt`
- `forward_tracker.py`
- `requirements-tracker.txt`
- `forward_watchlist.csv`
- `data/forward_signals.csv`
- `.github/workflows/forward_tracker.yml`

`README.md` 與 `升級步驟.txt` 建議一起更新。

## 第一次啟動每日自動追蹤

完成 Commit 後：

1. 打開 GitHub repository。
2. 點上方 **Actions**。
3. 左邊點 **Daily forward validation**。
4. 點 **Run workflow**。
5. 再按綠色 **Run workflow**。
6. 等工作流程完成後，回到 repository。
7. 打開 `data/forward_signals.csv`，應該會看到第一批每日快照。
8. 回到 Streamlit，重新整理後打開「前向驗證」。

如果 workflow 在最後 `git push` 顯示權限錯誤，可到：

**Repository → Settings → Actions → General → Workflow permissions → Read and write permissions → Save**

然後重新 Run workflow。

## FinMind Token（完全選用）

沒有 Token 仍可使用匿名免費模式與 Yahoo 備援。

若你已有 FinMind Token，請不要把 Token 寫進程式或 CSV。

### GitHub Actions 使用 Token

Repository → Settings → Secrets and variables → Actions → New repository secret

名稱固定：

`FINMIND_TOKEN`

值填你的 Token。

### Streamlit 網站使用 Token

Streamlit App settings → Secrets，加入：

```toml
FINMIND_TOKEN = "你的 token"
```

v6 會自動讀取；也仍保留側欄手動輸入。

## v6 的資料檔會不會包含密碼？

不會。

`data/forward_signals.csv` 只保存股票研究資料與模型結果，不保存 FinMind Token、GitHub Token 或 Streamlit Secrets。

## 第一次驗證建議

升級後先確認：

1. 網站標題顯示 v6。
2. 原本 v1–v5 功能都正常。
3. 出現「前向驗證」頁籤。
4. GitHub Actions 可以手動成功執行。
5. `data/forward_signals.csv` 出現最新交易日資料。
6. 前向驗證頁顯示最新快照日期與股票數。
7. 過 5 個交易日後，再確認 `ret_5d` 開始有值。

## 重要限制

- 前向驗證從 v6 上線後才有意義，不應事後把 v6 以前的日期補成「當時的預測」。
- 目前報酬以每日收盤資料進行研究；它不是實際成交紀錄。
- MFE / MAE 使用日 K 高低價，無法知道盤中先後順序。
- 0050 是研究基準，不代表適合所有策略或所有股票。
- 免費資料來源可能有延遲、缺漏或調整。
- GitHub Actions 的排程可能因平台負載而延遲，不保證精確在 17:30 開始。
- 歷史回測與前向驗證都不保證未來獲利。

本專案仍是教育與研究工具，不是投資建議或自動交易系統。
