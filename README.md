# 免費台股 AI 多空分析系統 v2

v2 是 v1 的直接升級版，維持零付費部署方向，新增：

- 四色力道 K：強多 / 偏多 / 偏空 / 強空
- V/A 轉折訊號：加入 MA20 + MACD 同向確認
- 趨勢翻轉線：EMA20 依多空狀態顯示
- Expanding-window Walk-forward：避免一般一次切分回測太樂觀
- Brier score：檢查機率預測誤差
- 回測交易成本假設可調整
- 免費「觀察池」選股排行，一次最多 25 檔
- 選股排行 CSV 下載

## 你已經有 v1：如何升級

最簡單的方法是把 GitHub repository 裡下面 3 個檔案覆蓋成 v2：

1. `streamlit_app.py`
2. `stock_engine.py`
3. `requirements.txt`

這版 requirements 與前版相容，但建議一起覆蓋，避免之後版本不一致。

Streamlit Community Cloud 連著 GitHub，commit 後通常會自動重新部署。

## 資料來源

1. FinMind `TaiwanStockPrice`
2. FinMind `TaiwanStockInfo`（股票名稱；若失敗使用內建名稱）
3. Yahoo Finance / yfinance 作為備援

## 模型說明

預測目標：未來 5 個交易日報酬是否 > 0。

v2 的 walk-forward 在預測每一段測試資料時，只使用當時已經能知道未來 5 日結果的更早資料。因為 5 日標籤本身會用到未來，因此訓練資料還會額外留出 5 個交易日，避免標籤偷看到預測日之後的價格。

模型：Random Forest。

注意：「AI 上漲機率」不是保證，也不是個別投資建議。

## 選股排行說明

為了維持免費方案，v2 不做全台股上千檔即時掃描，而採觀察池模式，一次最多 25 檔。這可以避免很快耗盡 FinMind 免費 API 額度與 Streamlit 免費主機資源。

技術排名分數：

- 60% 多空力道
- 20% 20 日動能
- 10% 量價方向
- 10% 20 日價格區間位置

它是排序研究工具，不是「最值得買」的評分。

## 免費部署

入口檔：`streamlit_app.py`

建議 Python：3.12
