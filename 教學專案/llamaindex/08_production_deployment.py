# 08_production_deployment.py - 生產環境部署與監控
import os
import logging
import time
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from typing import Dict, List, Any

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llamaindex_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionRAGSystem:
    """生產環境 RAG 系統"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化生產環境系統"""
        self.config = config
        self.index = None
        self.query_engine = None
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0.0
        }
        
        logger.info("初始化生產環境 RAG 系統...")
        self._setup_system()
    
    def _setup_system(self):
        """設定系統組件"""
        try:
            # 設定嵌入模型
            embed_model = OpenAIEmbedding(
                model=self.config.get("embedding_model", "text-embedding-3-small"),
                embed_batch_size=self.config.get("embed_batch_size", 10)
            )
            Settings.embed_model = embed_model
            
            # 設定節點解析器
            node_parser = SentenceSplitter(
                chunk_size=self.config.get("chunk_size", 1024),
                chunk_overlap=self.config.get("chunk_overlap", 200)
            )
            
            # 載入文件
            documents = self._load_documents()
            
            # 建立索引
            self.index = self._create_index(documents, node_parser)
            
            # 建立查詢引擎
            self.query_engine = self._create_query_engine()
            
            logger.info("系統設定完成")
            
        except Exception as e:
            logger.error(f"系統設定失敗: {e}")
            raise
    
    def _load_documents(self):
        """載入文件"""
        documents_dir = self.config.get("documents_dir", "sample_documents")
        
        if not os.path.exists(documents_dir):
            logger.warning(f"文件目錄不存在: {documents_dir}")
            return []
        
        reader = SimpleDirectoryReader(input_dir=documents_dir)
        documents = reader.load_data()
        
        logger.info(f"載入了 {len(documents)} 個文件")
        return documents
    
    def _create_index(self, documents, node_parser):
        """建立索引"""
        if self.config.get("use_chroma", False):
            return self._create_chroma_index(documents, node_parser)
        else:
            return self._create_simple_index(documents, node_parser)
    
    def _create_simple_index(self, documents, node_parser):
        """建立簡單索引"""
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[node_parser]
        )
        logger.info("簡單索引建立完成")
        return index
    
    def _create_chroma_index(self, documents, node_parser):
        """建立 ChromaDB 索引"""
        try:
            # 初始化 ChromaDB
            chroma_client = chromadb.PersistentClient(
                path=self.config.get("chroma_path", "./chroma_production")
            )
            
            collection = chroma_client.get_or_create_collection(
                name=self.config.get("collection_name", "production_kb"),
                metadata={"description": "生產環境知識庫"}
            )
            
            # 建立向量儲存
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 建立索引
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[node_parser]
            )
            
            logger.info(f"ChromaDB 索引建立完成，集合: {collection.name}")
            return index
            
        except Exception as e:
            logger.error(f"ChromaDB 索引建立失敗: {e}")
            return self._create_simple_index(documents, node_parser)
    
    def _create_query_engine(self):
        """建立查詢引擎"""
        query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.get("similarity_top_k", 3),
            response_mode=self.config.get("response_mode", "compact"),
            streaming=self.config.get("streaming", False)
        )
        
        logger.info("查詢引擎建立完成")
        return query_engine
    
    def query(self, question: str) -> Dict[str, Any]:
        """執行查詢"""
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        try:
            logger.info(f"執行查詢: {question}")
            
            response = self.query_engine.query(question)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 更新指標
            self.metrics["successful_queries"] += 1
            self._update_average_response_time(response_time)
            
            result = {
                "success": True,
                "response": response.response,
                "response_time": response_time,
                "source_nodes": len(response.source_nodes),
                "timestamp": time.time()
            }
            
            logger.info(f"查詢成功，回應時間: {response_time:.2f}秒")
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            self.metrics["failed_queries"] += 1
            logger.error(f"查詢失敗: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time,
                "timestamp": time.time()
            }
    
    def _update_average_response_time(self, response_time: float):
        """更新平均回應時間"""
        total_successful = self.metrics["successful_queries"]
        current_avg = self.metrics["average_response_time"]
        
        # 計算新的平均值
        new_avg = ((current_avg * (total_successful - 1)) + response_time) / total_successful
        self.metrics["average_response_time"] = new_avg
    
    def get_metrics(self) -> Dict[str, Any]:
        """獲取系統指標"""
        success_rate = 0
        if self.metrics["total_queries"] > 0:
            success_rate = self.metrics["successful_queries"] / self.metrics["total_queries"]
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "system_status": "healthy" if success_rate > 0.9 else "degraded"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """健康檢查"""
        try:
            # 執行簡單查詢測試
            test_result = self.query("測試查詢")
            
            return {
                "status": "healthy" if test_result["success"] else "unhealthy",
                "response_time": test_result.get("response_time", 0),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"健康檢查失敗: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

def demonstrate_production_setup():
    """示範生產環境設定"""
    print("🏭 生產環境設定示範...")
    
    # 生產環境配置
    production_config = {
        "embedding_model": "text-embedding-3-small",
        "embed_batch_size": 10,
        "chunk_size": 1024,
        "chunk_overlap": 200,
        "similarity_top_k": 3,
        "response_mode": "compact",
        "streaming": False,
        "use_chroma": True,
        "chroma_path": "./chroma_production",
        "collection_name": "production_kb",
        "documents_dir": "sample_documents"
    }
    
    # 建立生產環境系統
    rag_system = ProductionRAGSystem(production_config)
    
    return rag_system

def demonstrate_monitoring(rag_system):
    """示範監控功能"""
    print("\n📊 監控功能示範...")
    
    # 執行測試查詢
    test_queries = [
        "什麼是人工智慧？",
        "雲端運算的優勢有哪些？",
        "機器學習和深度學習的差異？"
    ]
    
    for query in test_queries:
        print(f"\n測試查詢: {query}")
        result = rag_system.query(query)
        
        if result["success"]:
            print(f"✅ 查詢成功")
            print(f"   回應時間: {result['response_time']:.2f}秒")
            print(f"   來源節點: {result['source_nodes']}個")
            print(f"   回應: {result['response'][:100]}...")
        else:
            print(f"❌ 查詢失敗: {result['error']}")
    
    # 顯示系統指標
    print(f"\n📈 系統指標:")
    metrics = rag_system.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # 健康檢查
    print(f"\n🏥 健康檢查:")
    health = rag_system.health_check()
    print(f"   狀態: {health['status']}")
    print(f"   回應時間: {health.get('response_time', 0):.2f}秒")

def demonstrate_error_handling():
    """示範錯誤處理"""
    print("\n🛡️ 錯誤處理示範...")
    
    # 建立系統
    config = {
        "documents_dir": "sample_documents",
        "use_chroma": False
    }
    
    rag_system = ProductionRAGSystem(config)
    
    # 測試各種錯誤情況
    error_tests = [
        "",  # 空查詢
        "a" * 10000,  # 過長查詢
        "測試查詢" * 100,  # 重複查詢
    ]
    
    for i, test_query in enumerate(error_tests, 1):
        print(f"\n錯誤測試 {i}: {test_query[:50]}...")
        result = rag_system.query(test_query)
        
        if result["success"]:
            print(f"✅ 處理成功")
        else:
            print(f"❌ 處理失敗: {result['error']}")

def demonstrate_scalability():
    """示範可擴展性"""
    print("\n📈 可擴展性示範...")
    
    # 建立系統
    config = {
        "documents_dir": "sample_documents",
        "use_chroma": True,
        "chroma_path": "./chroma_scalable"
    }
    
    rag_system = ProductionRAGSystem(config)
    
    # 並行查詢測試
    import concurrent.futures
    import threading
    
    def query_worker(query_id):
        """查詢工作函數"""
        query = f"測試查詢 {query_id}"
        result = rag_system.query(query)
        return result
    
    # 執行並行查詢
    print("執行並行查詢測試...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(query_worker, i) for i in range(10)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    successful_queries = sum(1 for r in results if r["success"])
    
    print(f"並行查詢結果:")
    print(f"   總查詢數: {len(results)}")
    print(f"   成功查詢數: {successful_queries}")
    print(f"   總時間: {total_time:.2f}秒")
    print(f"   平均時間: {total_time/len(results):.2f}秒/查詢")

def explain_production_considerations():
    """解釋生產環境考量"""
    print("\n📚 生產環境考量:")
    print("""
    1. 🏗️ 架構設計
       - 微服務架構
       - 負載均衡
       - 容錯機制
       - 水平擴展
    
    2. 💾 資料管理
       - 向量資料庫選擇
       - 資料備份策略
       - 索引更新機制
       - 資料一致性
    
    3. 🔒 安全性
       - API 認證授權
       - 資料加密
       - 輸入驗證
       - 存取控制
    
    4. 📊 監控與日誌
       - 效能指標追蹤
       - 錯誤日誌記錄
       - 健康檢查機制
       - 告警系統
    
    5. ⚡ 效能優化
       - 快取策略
       - 查詢優化
       - 資源管理
       - 並行處理
    
    6. 🔄 部署與維護
       - 容器化部署
       - 自動化部署
       - 版本管理
       - 回滾機制
    
    7. 📈 擴展性
       - 水平擴展
       - 垂直擴展
       - 負載分散
       - 資源調度
    """)

def demonstrate_best_practices():
    """示範最佳實踐"""
    print("\n✨ 最佳實踐示範...")
    
    # 最佳實踐配置
    best_practices_config = {
        "embedding_model": "text-embedding-3-small",
        "embed_batch_size": 20,  # 較大的批次大小
        "chunk_size": 1024,
        "chunk_overlap": 200,
        "similarity_top_k": 5,  # 更多的檢索結果
        "response_mode": "compact",
        "streaming": True,  # 啟用串流
        "use_chroma": True,
        "chroma_path": "./chroma_best_practices",
        "collection_name": "best_practices_kb",
        "documents_dir": "sample_documents"
    }
    
    # 建立最佳實踐系統
    rag_system = ProductionRAGSystem(best_practices_config)
    
    # 測試查詢
    query = "請詳細說明人工智慧的發展趨勢和未來展望"
    print(f"查詢: {query}")
    
    result = rag_system.query(query)
    
    if result["success"]:
        print(f"✅ 查詢成功")
        print(f"   回應時間: {result['response_time']:.2f}秒")
        print(f"   回應長度: {len(result['response'])}字元")
        print(f"   來源節點: {result['source_nodes']}個")
    else:
        print(f"❌ 查詢失敗: {result['error']}")

if __name__ == "__main__":
    try:
        print("🚀 開始學習生產環境部署...")
        
        # 解釋生產環境考量
        explain_production_considerations()
        
        # 示範生產環境設定
        rag_system = demonstrate_production_setup()
        
        # 示範監控功能
        demonstrate_monitoring(rag_system)
        
        # 示範錯誤處理
        demonstrate_error_handling()
        
        # 示範可擴展性
        demonstrate_scalability()
        
        # 示範最佳實踐
        demonstrate_best_practices()
        
        print("\n🎉 生產環境部署學習完成！")
        print("恭喜！您已經完成了完整的 LlamaIndex 學習路徑！")
        
    except Exception as e:
        logger.error(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()