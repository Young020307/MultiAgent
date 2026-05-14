# rag/retrieval.py
"""混合检索优化器：向量检索 + BM25 关键词检索 + RRF 重排。"""

import logging
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RRF_K = 60
VECTOR_K = 5
BM25_K = 5
HYBRID_K = 3

class RetrievalOptimizer:
    """混合检索优化器。

    并行执行向量检索 + BM25 关键词检索，使用 RRF 算法融合排序。

    用法:
        optimizer = RetrievalOptimizer(vectorstore, chunks)
        docs = optimizer.hybrid_search("查询文本", top_k=3)
    """

    def __init__(self, vectorstore: Chroma, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self._setup_retrievers()

    def _setup_retrievers(self):
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": VECTOR_K},
        )
        
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=BM25_K,
        )
        logger.info("检索器初始化完成: vector(top_k=5) + BM25(top_k=5)")

    def hybrid_search(self, query: str, top_k: int = HYBRID_K) -> List[Document]:
        """混合检索: 向量 + BM25 并行执行，RRF 融合后返回 top_k。"""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        print(f"[DEBUG] 向量检索召回 {len(vector_docs)} 条")
        print(f"[DEBUG] BM25 检索召回 {len(bm25_docs)} 条")
        ranked = self._rrf_rank(vector_docs, bm25_docs, k=RRF_K)
        return ranked[:top_k]

    @staticmethod
    def _rrf_rank(
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = RRF_K,
    ) -> List[Document]:
        """RRF 融合排序。

        score(d) = 1/(k + rank_vector(d)) + 1/(k + rank_bm25(d))
        """
        doc_scores: Dict[int, float] = {}
        doc_objects: Dict[int, Document] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for doc_id, score in sorted_items:
            doc = doc_objects[doc_id]
            doc.metadata["rrf_score"] = score
            result.append(doc)

        return result
