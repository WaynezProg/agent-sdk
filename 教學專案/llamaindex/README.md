# LlamaIndex 教學專案

## 專案概述

LlamaIndex 是一個強大的 RAG（Retrieval-Augmented Generation）框架，專門用於建立智能文件檢索和問答系統。本教學專案將帶您從基礎到進階，完整掌握 LlamaIndex 的核心功能。

## 學習路徑

### 基礎階段
1. **01_basic_setup.py** - 環境設定與基礎概念
2. **02_document_loading.py** - 文件載入與處理
3. **03_vector_index.py** - 向量索引建立
4. **04_query_retrieval.py** - 查詢與檢索

### 進階階段
5. **05_rag_integration.py** - RAG 系統整合
6. **06_agent_integration.py** - 與 Agent SDK 整合
7. **07_advanced_features.py** - 進階功能（多模態、自定義檢索器）
8. **08_production_deployment.py** - 生產環境部署

## 環境需求

```bash
pip install llama-index
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
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

- **文件處理**：PDF、Word、網頁等多種格式
- **向量化**：文字嵌入與相似性搜尋
- **檢索策略**：關鍵字、語義、混合檢索
- **RAG 架構**：檢索增強生成的最佳實踐
- **效能優化**：索引建立與查詢優化

## 與 Agent SDK 的整合

本教學專案特別設計了與 OpenAI Agent SDK 的整合範例，讓您能夠：
- 將 LlamaIndex 作為 Agent 的工具
- 建立智能文件問答系統
- 實現多代理協作的知識檢索

## 進階應用

- 多模態文件處理（圖片、音頻）
- 自定義檢索器開發
- 分散式索引與查詢
- 生產環境部署與監控