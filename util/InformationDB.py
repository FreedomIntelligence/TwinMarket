from typing import Optional, Dict
import os
import pandas as pd
import numpy as np
import faiss
from tqdm.auto import tqdm
import pickle
from pathlib import Path
import multiprocessing as mp
from typing import List, Dict, Optional
# import torch
import random
import requests
import yaml

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)


class EmbeddingWorker:
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
                "Content-Type": "application/json"
            }
            payload = {
                "input": text,
                "model": self.model_name
            }

            # Send request to the API
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload
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
    def __init__(self,
                 config_path: str = "config/embedding.yaml",
                 database_dir: str = "data/InformationDB",
                 max_workers: Optional[int] = None):
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
            embedding = self.worker.get_embedding(row['content'])
            if embedding is not None:
                metadata = {
                    'content': row['content'],
                    'title': row['title'],
                    'type': news_type,
                }
                # 根据不同新闻类型处理日期和其他字段
                if news_type == 'announcement':
                    metadata.update({
                        'datetime': row['ann_date'],
                        'ts_code': row['ts_code'],
                        'stock_name': row['name'],
                        'industry': row['industry'],
                    })
                elif news_type == 'cctv':
                    metadata.update({
                        'datetime': row['date'],
                    })
                else:  # long_news 和 short_news
                    metadata.update({
                        'datetime': row['datetime'],
                        'source': row['source']
                    })

                results.append((embedding, metadata))
        return results

    def process_file(self, file_path: str) -> Optional[List[tuple]]:
        # 从文件路径判断新闻类型
        df = pd.read_csv(file_path)
        if 'ann_' in file_path:
            news_type = 'announcement'
        elif 'cctv_news' in file_path:
            news_type = 'cctv'
        elif 'long_news' in file_path:
            news_type = 'long_news'
        elif 'short_news' in file_path:
            news_type = 'short_news'
        else:
            print(f"Unknown file type: {file_path}")
            return None

        # 转换为字典列表并处理
        records = df.to_dict('records')
        return self.process_batch(records, news_type)

    def build_database(self, data_path: str, batch_size: int = 32, folder_name: str = "2023_new"):
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
                            if file.startswith(('ann_', 'cctv_news_', 'long_news_', 'short_news_')):
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
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.index, f)
            with open(self.metadata_path, 'wb') as f:
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

            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False

    def search_announcements(self,
                             start_date: pd.Timestamp,
                             end_date: pd.Timestamp,
                             query: str,
                             ts_code: str = None,
                             top_k=3):
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
            if meta['type'] != 'announcement' or not (start_date <= pd.to_datetime(meta['datetime']) <= end_date):
                continue
            if ts_code is not None and meta['ts_code'] != ts_code:
                continue
            results.append({
                'distance': distances[0][i],
                'content': meta['content'],
                'title': meta['title'],
                'datetime': pd.to_datetime(meta['datetime'])
            })
            if len(results) >= top_k:
                break
        return results

    def search_news(self,
                    start_date: pd.Timestamp,
                    end_date: pd.Timestamp,
                    query: str,
                    top_k=3,
                    type=None):
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
            if (meta['type'] != 'announcement' and
                    start_date <= pd.to_datetime(meta['datetime']) <= end_date and
                    (type is None or meta['type'] == type)):
                results.append({
                    'distance': distances[0][i],
                    'content': meta['content'],
                    'title': meta['title'],
                    'datetime': pd.to_datetime(meta['datetime']),
                    'type': meta['type']
                })
                if len(results) >= top_k:
                    break
        return results

    def search_news_batch(self,
                          start_date: pd.Timestamp,
                          end_date: pd.Timestamp,
                          queries: List[str],
                          top_k: int = 3,
                          type: Optional[str] = None) -> List[List[Dict]]:
        """
        批量搜索新闻
        """
        if not queries or self.index is None:
            return []

        # 1. 将所有 query 转换为 embedding
        query_embs = [self.get_text_embedding(query) for query in queries]
        # 过滤掉 None 的embedding
        valid_queries_and_embs = [(q, emb) for q, emb in zip(queries, query_embs) if emb is not None]
        if not valid_queries_and_embs:
            return []
        valid_queries, query_embs = zip(*valid_queries_and_embs)
        query_embs = list(query_embs)

        # 如果没有embedding，直接返回
        if not query_embs:
            return []

        # 1.5 转换为numpy数组
        query_embs_np = np.array(query_embs, dtype=np.float32)
        query_embs_np = query_embs_np.reshape(query_embs_np.shape[0], -1)  # 将 (2, 1, 1024) 转换为 (2, 1024)

        # 2. 使用 Faiss 进行批量查询
        distances, indices = self.index.search(query_embs_np, 1000)

        # 3. 处理查询结果
        all_results = []
        for i, query in enumerate(valid_queries):
            results = []
            for j, idx in enumerate(indices[i]):
                meta = self.metadata[idx]
                if (meta['type'] != 'announcement' and
                        start_date <= pd.to_datetime(meta['datetime']) <= end_date and
                        (type is None or meta['type'] == type)):
                    results.append({
                        'distance': distances[i][j],
                        'content': meta['content'],
                        'title': meta['title'],
                        'datetime': pd.to_datetime(meta['datetime']),
                        'type': meta['type']
                    })
                    if len(results) >= top_k:
                        break
            all_results.append(results)
        return all_results
