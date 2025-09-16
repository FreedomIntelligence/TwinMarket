"""
智能信息检索数据库模块

该模块实现了一个基于向量相似性的智能信息检索系统，专门用于处理
新闻、公告等文本信息的高效检索和相关性分析。

核心功能：
- 文本向量化：使用API服务将文本转换为高维向量表示
- 向量数据库：基于FAISS的高性能向量相似性搜索
- 批量处理：支持大规模文本数据的并行处理
- 多类型支持：支持新闻、公告、CCTV新闻等多种信息类型
- 智能检索：基于语义相似性的精确信息检索

技术架构：
- 嵌入模型：支持多种API服务的文本嵌入
- 向量索引：FAISS高性能向量检索引擎
- 数据管理：完整的元数据管理和持久化
- 并发处理：多进程并行的数据处理能力
- 缓存机制：向量和元数据的高效缓存

数据类型支持：
- announcement：公司公告信息
- cctv：央视新闻内容
- long_news：长篇新闻报道
- short_news：短篇新闻快讯

适用场景：
- 投资信息检索
- 新闻相关性分析
- 公告影响评估
- 市场情报收集
- 智能投资助手
"""

# 标准库导入
import multiprocessing as mp
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

# 第三方库导入
import faiss
import numpy as np
import pandas as pd
import requests
import yaml
from tqdm.auto import tqdm

# 设置多进程启动方法
mp.set_start_method("spawn", force=True)


class EmbeddingWorker:
    """
    文本嵌入向量生成工作器

    该类负责调用外部API服务将文本转换为高维向量表示，
    是整个信息检索系统的基础组件。支持多API密钥的负载均衡。

    核心功能：
    - API配置管理：从YAML文件加载API配置信息
    - 多密钥支持：随机选择API密钥实现负载均衡
    - 文本向量化：将文本内容转换为数值向量
    - 错误处理：完善的API调用错误处理机制
    - 向量标准化：确保向量格式的一致性

    Attributes:
        config (Dict): API配置信息
        api_keys (List): API密钥列表
        model_name (str): 使用的嵌入模型名称
        base_url (str): API服务的基础URL

    Note:
        - 支持多个API密钥的随机选择
        - 自动处理向量维度和格式问题
        - 包含完整的异常处理和错误日志
    """

    def __init__(self, config_path: str = "config/embedding.yaml"):
        self.config = self._load_config(config_path)
        self.api_keys = self.config["api_key"]
        self.model_name = self.config["model_name"]
        self.base_url = self.config["base_url"]

    def _load_config(self, config_path: str) -> Dict:
        """Load API configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _get_random_api_key(self) -> str:
        """Randomly select an API key from the list."""
        return random.choice(self.api_keys)

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a given text using the API."""
        if not text or not isinstance(text, str):
            return None

        try:
            # Prepare API request payload
            headers = {
                "Authorization": f"Bearer {self._get_random_api_key()}",  # Randomly select a key for each request
                "Content-Type": "application/json",
            }
            payload = {"input": text, "model": self.model_name}

            # Send request to the API
            response = requests.post(
                f"{self.base_url}/embeddings", headers=headers, json=payload
            )
            response.raise_for_status()  # Raise an error for bad responses

            # Parse the response
            embedding_data = response.json()
            embedding = np.array(embedding_data["data"][0]["embedding"])

            # Ensure 2D array
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim > 2:
                embedding = embedding.squeeze()
                if embedding.ndim > 2:
                    raise ValueError(f"Invalid embedding shape: {embedding.shape}")
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)

            return embedding

        except Exception as e:
            print(f"Error generating embedding for text '{text[:100]}...': {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return None


class InformationDB:
    """
    智能信息检索数据库主类

    该类是整个信息检索系统的核心，集成了文本嵌入、向量索引、
    数据管理等功能，提供了完整的语义搜索能力。

    系统架构：
    1. 嵌入层：EmbeddingWorker负责文本向量化
    2. 索引层：FAISS向量索引提供高速相似性搜索
    3. 数据层：元数据管理和持久化存储
    4. 服务层：提供统一的检索接口

    核心特性：
    - 语义搜索：基于向量相似性的智能搜索
    - 多类型支持：支持多种新闻和公告类型
    - 批量操作：高效的批量检索和处理
    - 数据持久化：向量索引和元数据的持久化存储
    - 性能优化：多进程处理和缓存机制

    Attributes:
        worker (EmbeddingWorker): 文本嵌入工作器实例
        index (faiss.Index): FAISS向量索引
        metadata (List): 文档元数据列表
        database_dir (Path): 数据库文件目录
        max_workers (int): 最大并发工作进程数

    Example:
        >>> db = InformationDB("config/embedding.yaml", "data/news_db")
        >>> db.load_database()
        >>> results = db.search_news(start_date, end_date, "市场趋势", top_k=5)
    """

    def __init__(
        self,
        config_path: str = "config/embedding.yaml",
        database_dir: str = "data/InformationDB",
        max_workers: Optional[int] = None,
    ):
        self.worker = EmbeddingWorker(config_path)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index = None
        self.metadata = []
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers or mp.cpu_count()
        self.index_path = self.database_dir / "faiss_index.pkl"
        self.metadata_path = self.database_dir / "metadata.pkl"

    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        embedding = self.worker.get_embedding(text)
        return embedding

    def process_batch(self, batch: List[Dict], news_type: str) -> List[tuple]:
        results = []
        for row in tqdm(batch, desc="Processing items", leave=False):
            embedding = self.worker.get_embedding(row["content"])
            if embedding is not None:
                metadata = {
                    "content": row["content"],
                    "title": row["title"],
                    "type": news_type,
                }
                # 根据不同新闻类型处理日期和其他字段
                if news_type == "announcement":
                    metadata.update(
                        {
                            "datetime": row["ann_date"],
                            "ts_code": row["ts_code"],
                            "stock_name": row["name"],
                            "industry": row["industry"],
                        }
                    )
                elif news_type == "cctv":
                    metadata.update(
                        {
                            "datetime": row["date"],
                        }
                    )
                else:  # long_news 和 short_news
                    metadata.update(
                        {"datetime": row["datetime"], "source": row["source"]}
                    )

                results.append((embedding, metadata))
        return results

    def process_file(self, file_path: str) -> Optional[List[tuple]]:
        # 从文件路径判断新闻类型
        df = pd.read_csv(file_path)
        if "ann_" in file_path:
            news_type = "announcement"
        elif "cctv_news" in file_path:
            news_type = "cctv"
        elif "long_news" in file_path:
            news_type = "long_news"
        elif "short_news" in file_path:
            news_type = "short_news"
        else:
            print(f"Unknown file type: {file_path}")
            return None

        # 转换为字典列表并处理
        records = df.to_dict("records")
        return self.process_batch(records, news_type)

    def build_database(
        self, data_path: str, batch_size: int = 32, folder_name: str = "2023_new"
    ):
        if self.load_database():
            print("Loaded existing database from disk")
            return

        files = []
        for year_dir, year_subdirs, _ in os.walk(data_path):
            if os.path.basename(year_dir) == folder_name:
                for month_dir in year_subdirs:
                    month_path = os.path.join(year_dir, month_dir)
                    for date_dir, _, filenames in os.walk(month_path):
                        for file in filenames:
                            if file.startswith(
                                ("ann_", "cctv_news_", "long_news_", "short_news_")
                            ):
                                files.append((os.path.join(date_dir, file), batch_size))

        all_embeddings = []
        pbar = tqdm(total=len(files), desc="Processing files", position=0)

        for file_path, batch_size in files:
            file_results = self.process_file(file_path)
            if file_results:
                embeddings, metadata = zip(*file_results)
                all_embeddings.extend(embeddings)
                self.metadata.extend(metadata)
            pbar.update(1)

        pbar.close()

        if all_embeddings:
            print("Building FAISS index...")
            all_embeddings = np.vstack(all_embeddings)
            self.index = faiss.IndexFlatL2(all_embeddings.shape[1])
            self.index.add(all_embeddings)
            print("Saving database to disk...")
            self.save_database()

    def save_database(self):
        """Save the FAISS index and metadata to disk"""
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.index, f)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False

    def load_database(self):
        """Load the FAISS index and metadata from disk"""
        try:
            if not (self.index_path.exists() and self.metadata_path.exists()):
                return False

            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False

    def search_announcements(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        query: str,
        ts_code: str = None,
        top_k=3,
    ):
        """
        Search company announcements within a date range.
        """
        query_emb = self.get_text_embedding(query)
        if query_emb is None or self.index is None:
            return []
        distances, indices = self.index.search(query_emb, self.index.ntotal)
        results = []
        for i, idx in enumerate(indices[0]):
            meta = self.metadata[idx]
            if meta["type"] != "announcement" or not (
                start_date <= pd.to_datetime(meta["datetime"]) <= end_date
            ):
                continue
            if ts_code is not None and meta["ts_code"] != ts_code:
                continue
            results.append(
                {
                    "distance": distances[0][i],
                    "content": meta["content"],
                    "title": meta["title"],
                    "datetime": pd.to_datetime(meta["datetime"]),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def search_news(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        query: str,
        top_k=3,
        type=None,
    ):
        """
        Search news articles within a date range.
        """
        query_emb = self.get_text_embedding(query)
        if query_emb is None or self.index is None:
            return []
        distances, indices = self.index.search(query_emb, self.index.ntotal)
        results = []
        for i, idx in enumerate(indices[0]):
            meta = self.metadata[idx]
            if (
                meta["type"] != "announcement"
                and start_date <= pd.to_datetime(meta["datetime"]) <= end_date
                and (type is None or meta["type"] == type)
            ):
                results.append(
                    {
                        "distance": distances[0][i],
                        "content": meta["content"],
                        "title": meta["title"],
                        "datetime": pd.to_datetime(meta["datetime"]),
                        "type": meta["type"],
                    }
                )
                if len(results) >= top_k:
                    break
        return results

    def search_news_batch(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        queries: List[str],
        top_k: int = 3,
        type: Optional[str] = None,
    ) -> List[List[Dict]]:
        """
        批量新闻检索功能 - 高效的多查询并行处理

        该函数实现了高性能的批量新闻检索，能够同时处理多个查询请求，
        显著提高检索效率。特别适用于需要同时查询多个关键词的场景。

        性能优化特性：
        1. 批量向量化：一次性处理多个查询的向量转换
        2. 并行检索：使用FAISS的批量搜索能力
        3. 结果分组：自动将结果按查询分组返回
        4. 内存优化：高效的向量数组处理
        5. 容错处理：自动过滤无效查询和结果

        Args:
            start_date (pd.Timestamp): 检索开始日期
            end_date (pd.Timestamp): 检索结束日期
            queries (List[str]): 查询关键词列表
            top_k (int): 每个查询返回的最大结果数，默认3个
            type (Optional[str]): 指定新闻类型过滤，可选

        Returns:
            List[List[Dict]]: 批量检索结果，外层列表对应查询，内层列表包含：
                - distance: 相似性距离（越小越相似）
                - content: 新闻内容
                - title: 新闻标题
                - datetime: 发布时间
                - type: 新闻类型

        Note:
            - 查询顺序与结果顺序一一对应
            - 自动过滤无效的嵌入向量
            - 支持灵活的新闻类型过滤
            - 批量处理显著提高性能
        """
        if not queries or self.index is None:
            return []

        # 1. 将所有 query 转换为 embedding
        query_embs = [self.get_text_embedding(query) for query in queries]
        # 过滤掉 None 的embedding
        valid_queries_and_embs = [
            (q, emb) for q, emb in zip(queries, query_embs) if emb is not None
        ]
        if not valid_queries_and_embs:
            return []
        valid_queries, query_embs = zip(*valid_queries_and_embs)
        query_embs = list(query_embs)

        # 如果没有embedding，直接返回
        if not query_embs:
            return []

        # 1.5 转换为numpy数组
        query_embs_np = np.array(query_embs, dtype=np.float32)
        query_embs_np = query_embs_np.reshape(
            query_embs_np.shape[0], -1
        )  # 将 (2, 1, 1024) 转换为 (2, 1024)

        # 2. 使用 Faiss 进行批量查询
        distances, indices = self.index.search(query_embs_np, 1000)

        # 3. 处理查询结果
        all_results = []
        for i, query in enumerate(valid_queries):
            results = []
            for j, idx in enumerate(indices[i]):
                meta = self.metadata[idx]
                if (
                    meta["type"] != "announcement"
                    and start_date <= pd.to_datetime(meta["datetime"]) <= end_date
                    and (type is None or meta["type"] == type)
                ):
                    results.append(
                        {
                            "distance": distances[i][j],
                            "content": meta["content"],
                            "title": meta["title"],
                            "datetime": pd.to_datetime(meta["datetime"]),
                            "type": meta["type"],
                        }
                    )
                    if len(results) >= top_k:
                        break
            all_results.append(results)
        return all_results
