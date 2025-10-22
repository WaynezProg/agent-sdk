# AI 代理與 RAG 系統完整教學專案

## 專案概述

本專案提供完整的 AI 代理開發和 RAG（Retrieval-Augmented Generation）系統學習路徑，包含三個核心技術：

1. **LangChain** - 強大的 LLM 應用開發框架
2. **LlamaIndex** - 專業的 RAG 系統框架
3. **OpenAI Agent SDK** - 智能代理開發框架

## 專案結構

```
教學專案/
├── langchain/              # LangChain 教學
│   ├── 01_basic_setup.py
│   ├── 02_llm_models.py
│   ├── 03_prompts_templates.py
│   ├── 04_chains.py
│   ├── 05_memory.py
│   ├── 06_agents.py
│   ├── 07_tools.py
│   ├── 08_retrieval.py
│   ├── 09_embeddings.py
│   ├── 10_advanced_features.py
│   ├── 11_agent_integration.py
│   ├── 12_llamaindex_integration.py
│   ├── 13_production_deployment.py
│   ├── README.md
│   ├── 教學檔.md
│   └── requirements.txt
├── llamaindex/             # LlamaIndex 教學
│   ├── 01_basic_setup.py
│   ├── 02_document_loading.py
│   ├── 03_vector_index.py
│   ├── 04_query_retrieval.py
│   ├── 05_rag_integration.py
│   ├── 06_agent_integration.py
│   ├── 07_advanced_features.py
│   ├── 08_production_deployment.py
│   ├── README.md
│   ├── 教學檔.md
│   └── requirements.txt
├── agent_sdk/              # Agent SDK 教學
│   ├── 教學範例/
│   │   ├── 01_hello_agent.py
│   │   ├── 02_tools_math.py
│   │   ├── 03_tools_memory_todo.py
│   │   ├── 04_session_sqlite.py
│   │   ├── 05_handoffs.py
│   │   ├── 06_guardrail_min.py
│   │   ├── 07_structured_output.py
│   │   └── 08_file_search.py
│   ├── README.md
│   ├── 教學檔.md
│   └── requirements.txt
└── README.md               # 本文件
```

## 學習路徑建議

### 方案一：先學 LangChain，再學 LlamaIndex，最後學 Agent SDK
**適合對象：** 想要系統性學習 LLM 應用的學習者

1. **LangChain 基礎**（2-3 週）
   - 學習 LLM 應用開發基礎
   - 掌握提示詞工程
   - 了解鏈式處理
   - 學習記憶機制

2. **LlamaIndex 基礎**（1-2 週）
   - 學習 RAG 系統概念
   - 掌握文件處理
   - 了解向量索引
   - 學習查詢檢索

3. **Agent SDK 基礎**（1-2 週）
   - 學習代理開發
   - 掌握工具整合
   - 了解多代理協作

4. **整合應用**（1-2 週）
   - 建立完整的智能系統
   - 實現生產級應用

### 方案二：先學 LlamaIndex，再學 LangChain，最後學 Agent SDK
**適合對象：** 想要先掌握 RAG 系統的學習者

1. **LlamaIndex 基礎**（1-2 週）
   - 學習文件處理和索引
   - 掌握查詢檢索
   - 了解 RAG 系統架構

2. **LangChain 基礎**（2-3 週）
   - 學習 LLM 應用開發
   - 掌握複雜工作流程
   - 了解代理和工具

3. **Agent SDK 基礎**（1-2 週）
   - 學習代理開發
   - 掌握工具整合
   - 了解多代理協作

4. **整合應用**（1-2 週）
   - 建立智能知識助手
   - 實現多代理 RAG 系統

### 方案三：並行學習
**適合對象：** 有經驗的開發者

1. **基礎概念**（1-2 週）
   - 同時學習三個技術的基本概念
   - 了解各自的應用場景

2. **深度學習**（3-4 週）
   - 深入學習各自的核心功能
   - 完成所有教學範例

3. **整合專案**（2-3 週）
   - 建立完整的智能系統
   - 實現生產級應用

## 技術對比

| 特性 | LangChain | LlamaIndex | Agent SDK |
|------|-----------|------------|-----------|
| **主要用途** | LLM 應用開發框架 | RAG 系統建立 | 智能代理開發 |
| **核心功能** | 鏈式處理、代理、工具 | 文件檢索、向量搜尋 | 工具使用、多代理協作 |
| **適用場景** | 複雜 AI 應用、工作流程 | 知識問答、文件分析 | 任務執行、對話系統 |
| **學習難度** | 中等 | 中等 | 中等 |
| **整合性** | 高（豐富的生態系統） | 高（專注 RAG） | 高（可整合各種工具） |
| **生態系統** | 非常豐富 | 中等 | 中等 |
| **生產就緒** | 是 | 是 | 是 |

## 環境設定

### 1. 安裝依賴

```bash
# LangChain 依賴
cd langchain
pip install -r requirements.txt

# LlamaIndex 依賴
cd ../llamaindex
pip install -r requirements.txt

# Agent SDK 依賴
cd ../agent_sdk
pip install -r requirements.txt
```

### 2. 環境變數設定

在專案根目錄創建 `.env` 檔案：

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 快速開始

```bash
# 測試 LangChain
cd langchain
python 01_basic_setup.py

# 測試 LlamaIndex
cd ../llamaindex
python 01_basic_setup.py

# 測試 Agent SDK
cd ../agent_sdk/教學範例
python 01_hello_agent.py
```

## 整合應用範例

### 智能知識助手
結合 LangChain、LlamaIndex 和 Agent SDK，建立能夠：
- 使用 LangChain 建立複雜的問答流程
- 使用 LlamaIndex 提供知識庫檢索
- 使用 Agent SDK 執行具體任務
- 支援多輪對話和智能分析

### 多代理 RAG 系統
建立多個專業化代理：
- **搜尋代理**：使用 LlamaIndex 負責文件檢索
- **分析代理**：使用 LangChain 負責內容分析
- **回答代理**：使用 Agent SDK 負責生成回答
- **協調代理**：使用 LangChain 負責任務分配

### 複雜工作流程系統
- 使用 LangChain 編排整個工作流程
- 使用 LlamaIndex 提供知識支援
- 使用 Agent SDK 執行具體任務
- 實現端到端的智能處理

## 學習資源

### 官方文件
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [OpenAI Agent SDK](https://openai.github.io/openai-agents-python/)

### 進階學習
- 多模態 RAG 系統
- 分散式代理架構
- 生產環境部署
- 效能優化技巧

## 常見問題

### Q1: 應該先學哪個技術？
**建議：** 根據您的應用需求選擇：
- 如果需要建立複雜的 LLM 應用，先學 LangChain
- 如果需要建立知識問答系統，先學 LlamaIndex
- 如果需要建立對話系統或任務執行代理，先學 Agent SDK

### Q2: 三個技術可以獨立使用嗎？
**答案：** 可以，但整合使用效果更佳：
- LangChain 提供 LLM 應用開發框架
- LlamaIndex 提供知識檢索能力
- Agent SDK 提供智能代理能力
- 結合使用可以建立更強大的 AI 系統

### Q3: 學習需要多長時間？
**建議：** 
- 基礎學習：3-4 週
- 完整掌握：6-8 週
- 生產應用：8-12 週

### Q4: 需要什麼技術背景？
**要求：**
- Python 基礎
- 基本的機器學習概念
- API 使用經驗
- 資料庫基礎知識

## 專案貢獻

歡迎提交 Issue 和 Pull Request 來改進教學內容！

## 授權

本專案採用 MIT 授權。

## 聯絡資訊

如有問題或建議，請透過以下方式聯絡：
- 提交 GitHub Issue
- 發送郵件至 [your-email@example.com]

---

**祝您學習愉快！** 🚀

通過本教學專案，您將能夠掌握 LangChain、LlamaIndex 和 Agent SDK 三個強大的 AI 技術，建立完整的智能應用系統，為您的應用帶來智能化的能力。