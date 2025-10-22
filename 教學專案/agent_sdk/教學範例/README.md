# Agent SDK 教學範例

這個資料夾包含了 OpenAI Agents SDK 的完整學習範例，從基礎到進階功能。

## 範例說明

### 01_hello_agent.py
- **功能**：最基礎的 Agent 建立與執行
- **學習重點**：Agent 基本概念、Runner 使用方式
- **執行方式**：`python 01_hello_agent.py`

### 02_tools_math.py
- **功能**：示範如何建立和使用 function_tool
- **學習重點**：工具定義、函式裝飾器、數學運算
- **執行方式**：`python 02_tools_math.py`

### 03_tools_memory_todo.py
- **功能**：結合工具使用與記憶功能的待辦事項管理
- **學習重點**：工具組合、記憶機制、實用應用
- **執行方式**：`python 03_tools_memory_todo.py`

### 04_session_sqlite.py
- **功能**：SQLiteSession 資料庫記憶功能
- **學習重點**：會話管理、資料持久化、SQLite 整合
- **執行方式**：`python 04_session_sqlite.py`

### 05_handoffs.py
- **功能**：多代理協作與任務移交
- **學習重點**：handoff 機制、代理分工、協作架構
- **執行方式**：`python 05_handoffs.py`

### 06_guardrail_min.py
- **功能**：輸入輸出安全護欄
- **學習重點**：安全機制、內容過濾、tripwire 觸發
- **執行方式**：`python 06_guardrail_min.py`

### 07_structured_output.py
- **功能**：結構化輸出與 Pydantic 整合
- **學習重點**：型別定義、JSON Schema、資料驗證
- **執行方式**：`python 07_structured_output.py`

## 執行前準備

1. **安裝依賴**：
   ```bash
   pip install -r ../requirements.txt
   ```

2. **設定環境變數**：
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

3. **執行範例**：
   ```bash
   python 01_hello_agent.py
   ```

## 學習順序建議

建議按照檔案編號順序學習：
1. 先從 `01_hello_agent.py` 開始了解基礎概念
2. 學習 `02_tools_math.py` 掌握工具使用
3. 通過 `03_tools_memory_todo.py` 了解實用應用
4. 學習 `04_session_sqlite.py` 掌握記憶機制
5. 通過 `05_handoffs.py` 了解多代理協作
6. 學習 `06_guardrail_min.py` 掌握安全機制
7. 最後通過 `07_structured_output.py` 了解結構化輸出

## 注意事項

- 每個範例都可以獨立執行
- 部分範例會產生資料庫檔案，已加入 `.gitignore`
- 詳細教學內容請參考根目錄的 `教學檔.md`
