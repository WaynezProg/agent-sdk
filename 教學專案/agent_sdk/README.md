# Agent SDK 教學專案

## 專案結構

```
agent sdk/
├── 教學範例/              # 課程範例檔案
│   ├── 01_hello_agent.py
│   ├── 02_tools_math.py
│   ├── 03_tools_memory_todo.py
│   ├── 04_session_sqlite.py
│   ├── 05_handoffs.py
│   ├── 06_guardrail_min.py
│   └── 07_structured_output.py
├── db/                   # 資料庫檔案目錄（Git 忽略）
├── 教學檔.md              # 詳細教學文件
├── README.md              # 專案說明
└── requirements.txt       # 依賴套件
```

## 環境變數設定

本專案使用 `.env` 檔案來管理環境變數，避免每次都需要手動 export。

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 使用方式

1. 確保 `.env` 檔案存在於專案根目錄
2. 在 `.env` 檔案中設定您的環境變數：
   ```
   openaikey=your_api_key_here
   ```
3. 執行程式時，環境變數會自動載入：
   ```bash
   python 教學範例/01_hello_agent.py
   ```

### 最佳實踐

1. **安全性**：`.env` 檔案已加入 `.gitignore`，不會被提交到版本控制
2. **多環境支援**：可以創建不同環境的 `.env` 檔案：
   - `.env.development`
   - `.env.production`
   - `.env.test`
3. **環境變數命名**：建議使用大寫字母和下劃線，如 `OPENAI_API_KEY`

### 注意事項

- 請勿將包含真實 API key 的 `.env` 檔案提交到 Git
- 在團隊協作時，請提供 `.env.example` 檔案作為範本
- 定期更換 API key 以確保安全性