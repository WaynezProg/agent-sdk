# LangChain 教學專案

## 專案概述

LangChain 是一個強大的 LLM 應用開發框架，提供了豐富的工具和組件來構建複雜的 AI 應用。本教學專案將帶您從基礎到進階，完整掌握 LangChain 的核心功能。

## 學習路徑

### 基礎階段
1. **01_basic_setup.py** - 環境設定與基礎概念
2. **02_llm_models.py** - LLM 模型使用
3. **03_prompts_templates.py** - 提示詞模板
4. **04_chains.py** - 鏈式處理
5. **05_memory.py** - 記憶機制

### 進階階段
6. **06_agents.py** - 智能代理
7. **07_tools.py** - 工具整合
8. **08_retrieval.py** - 檢索增強生成 (RAG)
9. **09_embeddings.py** - 嵌入向量
10. **10_advanced_features.py** - 進階功能

### 整合應用
11. **11_agent_integration.py** - 與 Agent SDK 整合
12. **12_llamaindex_integration.py** - 與 LlamaIndex 整合
13. **13_production_deployment.py** - 生產環境部署

## 環境需求

```bash
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install python-dotenv
```

## 環境變數設定

在專案根目錄創建 `.env` 檔案：

```
OPENAI_API_KEY=your_openai_api_key_here
```

## 快速開始

1. 安裝依賴套件
2. 設定環境變數
3. 按照編號順序執行範例檔案

```bash
python 01_basic_setup.py
```

## 學習重點

- **LLM 整合**：OpenAI、Anthropic、本地模型
- **提示詞工程**：模板、變數、格式化
- **鏈式處理**：串聯多個 LLM 調用
- **記憶管理**：會話記憶、長期記憶
- **智能代理**：工具使用、決策邏輯
- **RAG 系統**：文件檢索、知識增強
- **向量資料庫**：嵌入、相似性搜尋

## 與其他技術的整合

本教學專案特別設計了與其他技術的整合範例：

- **Agent SDK 整合**：將 LangChain 作為 Agent 的工具
- **LlamaIndex 整合**：結合兩個 RAG 框架的優勢
- **多框架協作**：建立更強大的 AI 應用

## 進階應用

- 多模態處理（文字、圖片、音頻）
- 自定義工具開發
- 分散式處理
- 生產環境部署與監控

## 技術對比

| 特性 | LangChain | LlamaIndex | Agent SDK |
|------|-----------|------------|-----------|
| **主要用途** | LLM 應用開發框架 | RAG 系統建立 | 智能代理開發 |
| **核心功能** | 鏈式處理、代理、工具 | 文件檢索、向量搜尋 | 工具使用、多代理協作 |
| **適用場景** | 複雜 AI 應用 | 知識問答系統 | 任務執行、對話系統 |
| **學習難度** | 中等 | 中等 | 中等 |
| **整合性** | 高（豐富的生態系統） | 高（專注 RAG） | 高（可整合各種工具） |