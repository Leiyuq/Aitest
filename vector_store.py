import os
import json
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from oss2.exceptions import NoSuchKey
from utils import tokenize, clean_query
from config import AppConfig
#================================向量存储====================
class SimpleVectorStore:
    def __init__(self, store_dir: str, oss_bucket, oss_prefix: str):
        """
              store_dir: 本地临时目录，用于存放下载的索引文件
              oss_bucket: OSS Bucket 对象
              oss_prefix: OSS 中该项目的向量存储前缀（如 "knowledge_projects/项目名/vector_store"）
        """
        self.store_dir = store_dir
        self.oss_bucket = oss_bucket
        self.oss_prefix = oss_prefix
        Path(store_dir).mkdir(parents=True, exist_ok=True)

        self.index_file = os.path.join(store_dir, "index.json")
        self.vectors_file = os.path.join(store_dir, "vectors.npy")
        self.metadata_file = os.path.join(store_dir, "metadata.pkl")
        self.documents = []
        self.vocab = {}
        self.idf = {}
        self.vectors = None

    def _oss_key(self, filename: str) -> str:
        """生成 OSS 对象键"""
        return f"{self.oss_prefix}/{filename}"

    def _download_from_oss(self, local_path: str, object_key: str) -> bool:
        """从 OSS 下载文件到本地，如果不存在返回 False"""
        try:
            self.oss_bucket.get_object_to_file(object_key, local_path)
            return True
        except NoSuchKey:
            return False

    def _upload_to_oss(self, local_path: str, object_key: str):
        """上传本地文件到 OSS"""
        self.oss_bucket.put_object_from_file(object_key, local_path)

    def _delete_oss_object(self, object_key: str):
        """删除 OSS 对象"""
        try:
            self.oss_bucket.delete_object(object_key)
        except Exception:
            pass

    def _build_vocab_and_idf(self, all_texts: List[str]):
        doc_count = {}
        for text in all_texts:
            unique_words = set(tokenize(text))
            for w in unique_words:
                doc_count[w] = doc_count.get(w, 0) + 1

        total_docs = len(all_texts)
        self.vocab = {}
        self.idf = {}
        for w, df in doc_count.items():
            if 0.01 < df / total_docs < 0.8:
                idx = len(self.vocab)
                self.vocab[w] = idx
                self.idf[w] = np.log((total_docs + 1) / (df + 0.5))

    def _text_to_vector(self, text: str) -> np.ndarray:
        words = tokenize(text)
        if not words or not self.vocab:
            return np.zeros(len(self.vocab))
        tf = {}
        for w in words:
            if w in self.vocab:
                tf[w] = tf.get(w, 0) + 1
        vec = np.zeros(len(self.vocab))
        for w, cnt in tf.items():
            idx = self.vocab[w]
            tf_val = np.log1p(cnt / len(words))
            vec[idx] = tf_val * self.idf.get(w, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def build_index(self, documents: List[Dict], files_info: List[Dict] = None) -> bool:
        try:
            if not documents:
                return False
            self.documents = [{"id": i, "text": d["content"], "metadata": d.get("metadata", {})}
                              for i, d in enumerate(documents)]
            texts = [d["text"] for d in self.documents]
            self._build_vocab_and_idf(texts)
            vectors = [self._text_to_vector(t) for t in texts]
            self.vectors = np.array(vectors)
            # 保存到本地临时文件
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": self.documents,
                    "vocab": self.vocab,
                    "idf": self.idf
                }, f, ensure_ascii=False)
            np.save(self.vectors_file, self.vectors)
            if files_info:
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump({"files_info": files_info, "updated": datetime.now().isoformat()}, f)
            # 上传到 OSS
            self._upload_to_oss(self.index_file, self._oss_key("index.json"))
            self._upload_to_oss(self.vectors_file, self._oss_key("vectors.npy"))
            if files_info and os.path.exists(self.metadata_file):
                self._upload_to_oss(self.metadata_file, self._oss_key("metadata.pkl"))
            return True
        except Exception as e:
            st.error(f"构建索引失败: {e}")
            return False

    def load_index(self) -> bool:
        try:
            if not self._download_from_oss(self.index_file, self._oss_key("index.json")):
                return False
            if not self._download_from_oss(self.vectors_file, self._oss_key("vectors.npy")):
                return False
            self._download_from_oss(self.metadata_file, self._oss_key("metadata.pkl"))

            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.vocab = data["vocab"]
                self.idf = data["idf"]
            self.vectors = np.load(self.vectors_file)
            return True
        except Exception as e:
            st.warning(f"加载索引失败: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.vectors is None or len(self.documents) == 0:
            return []
        cleaned_query = clean_query(query)
        if not cleaned_query:
            cleaned_query = query
        q_vec = self._text_to_vector(cleaned_query)
        scores = np.dot(self.vectors, q_vec)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_idx:
            sim = float(scores[idx])
            if sim >= AppConfig.RAG_SIMILARITY_THRESHOLD:
                doc = self.documents[idx]
                results.append({
                    "content": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity": sim
                })
        return results

    def is_index_valid(self, current_files_info: List[Dict]) -> bool:
        """检查 OSS 上的元数据文件是否与当前文件列表一致"""
        temp_meta = os.path.join(self.store_dir, "metadata_check.pkl")
        if not self._download_from_oss(temp_meta, self._oss_key("metadata.pkl")):
            return False
        try:
            with open(temp_meta, 'rb') as f:
                meta = pickle.load(f)
            saved = {f["name"]: f for f in meta["files_info"]}
            curr = {f["name"]: f for f in current_files_info}
            if set(saved.keys()) != set(curr.keys()):
                return False
            for name, cf in curr.items():
                sf = saved[name]
                if sf.get("size_bytes") != cf.get("size_bytes") or sf.get("mtime") != cf.get("mtime"):
                    return False
            return True
        finally:
            if os.path.exists(temp_meta):
                os.remove(temp_meta)

    def get_built_files(self) -> set:
        if not self.documents:
            return set()
        return {doc["metadata"].get("source") for doc in self.documents if "source" in doc["metadata"]}

    def get_stats(self) -> Dict:
        return {
            "documents": len(self.documents),
            "vocab_size": len(self.vocab),
            "has_index": self.vectors is not None
        }

    def remove_file(self, filename: str) -> bool:
        """从当前内存索引中删除指定文件的分块，并重建索引，然后上传到 OSS"""
        if not self.documents:
            return False
        new_docs = [doc for doc in self.documents if doc["metadata"].get("source") != filename]
        if len(new_docs) == len(self.documents):
            return False
        if not new_docs:
            self.documents = []
            self.vocab = {}
            self.idf = {}
            self.vectors = None
            for fname in ["index.json", "vectors.npy", "metadata.pkl"]:
                self._delete_oss_object(self._oss_key(fname))
            for f in [self.index_file, self.vectors_file, self.metadata_file]:
                if os.path.exists(f):
                    os.remove(f)
            return True
        # 重建索引
        texts = [doc["text"] for doc in new_docs]
        self._build_vocab_and_idf(texts)
        vectors = [self._text_to_vector(t) for t in texts]
        self.vectors = np.array(vectors)
        self.documents = new_docs
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump({
                "documents": self.documents,
                "vocab": self.vocab,
                "idf": self.idf
            }, f, ensure_ascii=False)
        np.save(self.vectors_file, self.vectors)
        self._upload_to_oss(self.index_file, self._oss_key("index.json"))
        self._upload_to_oss(self.vectors_file, self._oss_key("vectors.npy"))
        return True

    def clear_index(self):
        self.documents = []
        self.vocab = {}
        self.idf = {}
        self.vectors = None
        for fname in ["index.json", "vectors.npy", "metadata.pkl"]:
            self._delete_oss_object(self._oss_key(fname))
        for f in [self.index_file, self.vectors_file, self.metadata_file]:
            if os.path.exists(f):
                os.remove(f)

    def update_metadata(self, files_info: List[Dict]):
        """单独更新 metadata.pkl 到 OSS"""
        if not files_info:
            self._delete_oss_object(self._oss_key("metadata.pkl"))
            return
        with open(self.metadata_file, 'wb') as f:
            pickle.dump({"files_info": files_info, "updated": datetime.now().isoformat()}, f)
        self._upload_to_oss(self.metadata_file, self._oss_key("metadata.pkl"))