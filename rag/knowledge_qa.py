# rag/knowledge_qa.py
"""Agentic RAG 引擎：查询改写 → 多查询并行检索 → 归一化 → JSON 返回。"""

import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.retrieval import RetrievalOptimizer

REWRITE_PROMPT = PromptTemplate(
    template="""你是一个查询改写专家。根据对话历史，将用户问题改写为1-3个适合向量检索的独立查询。
每行一个查询，不要序号，不要额外解释。
如果原问题已经清晰明确，直接原样返回即可。

对话历史：{history}
用户问题：{query}""",
    input_variables=["history", "query"],
)


class RAG:
    """Agentic RAG 引擎：自主查询改写 + 多查询检索 + 归一化 + JSON 返回。

    search(query, history="") → JSON str  完整管线，供 Agent 工具链使用
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "lrs33/bce-embedding-base_v1:latest",
        rewrite_model: str = "qwen2.5:3b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        top_k: int = 3,
    ):
        self.top_k = top_k

        print("初始化嵌入模型...")
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url,
        )

        print("连接向量库...")
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="customer_service_knowledge",
        )

        print("构建混合检索器...")
        all_data = self.vector_store.get(include=["documents", "metadatas"])
        chunks = []
        for i in range(len(all_data.get("ids", []))):
            chunks.append(Document(
                page_content=all_data["documents"][i],
                metadata=all_data["metadatas"][i] if all_data.get("metadatas") else {},
            ))
        self.optimizer = RetrievalOptimizer(self.vector_store, chunks)

        print("初始化查询改写 LLM...")
        self.rewrite_llm = ChatOllama(
            model=rewrite_model,
            base_url=ollama_base_url,
            temperature=0.1,
        )
        self.rewrite_chain = REWRITE_PROMPT | self.rewrite_llm | StrOutputParser()

    # ================================================================
    # 查询改写
    # ================================================================

    def _rewrite_query(self, query: str, history: str) -> list[str]:
        """生成 1-3 个改写查询，与原查询去重后返回。"""
        try:
            raw = self.rewrite_chain.invoke({"history": history, "query": query})
            rewritten = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        except Exception:
            rewritten = []

        seen = {query}
        result = [query]
        for q in rewritten:
            if q not in seen:
                result.append(q)
                seen.add(q)
        print(f"[RAG] 查询改写: {len(result)} 个查询 → {result}")
        return result

    # ================================================================
    # 多查询检索 + 去重
    # ================================================================

    def _multi_search(
        self, queries: list[str], top_k_per_query: int = 3
    ) -> list[Document]:
        """并行执行多个查询的混合检索，按 page_content hash 去重。"""
        from concurrent.futures import ThreadPoolExecutor

        seen: dict[int, Document] = {}
        all_docs = []
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [executor.submit(self.optimizer.hybrid_search, q, top_k=top_k_per_query)
                       for q in queries]
            for f in futures:
                all_docs.extend(f.result())

        for doc in all_docs:
            doc_id = hash(doc.page_content)
            if doc_id not in seen:
                seen[doc_id] = doc
            else:
                existing_score = seen[doc_id].metadata.get("rrf_score", 0)
                new_score = doc.metadata.get("rrf_score", 0)
                if new_score > existing_score:
                    seen[doc_id] = doc
        results = list(seen.values())
        print(f"[RAG] 多查询检索: {len(queries)} 查询 → {len(seen)} 个去重文档")
        return results

    # ================================================================
    # 分数归一化
    # ================================================================

    @staticmethod
    def _normalize_scores(docs: list[Document]) -> tuple[list[dict], float]:
        """RRF 分数归一化到 0-1，返回 (results_list, max_score)。"""
        if not docs:
            return [], 0.0

        max_rrf = max(doc.metadata.get("rrf_score", 0) for doc in docs)
        if max_rrf == 0:
            return [], 0.0

        results = []
        for doc in docs:
            score = doc.metadata.get("rrf_score", 0) / max_rrf
            results.append({
                "text": doc.page_content,
                "score": round(score, 4),
                "metadata": {k: v for k, v in doc.metadata.items() if k != "rrf_score"},
            })

        max_score = max(r["score"] for r in results)
        return results, max_score

    # ================================================================
    # 主入口
    # ================================================================

    def search(self, query: str, history: str = "") -> str:
        """完整管线：查询改写 → 多查询检索 → 归一化 → JSON 返回。"""
        queries = self._rewrite_query(query, history)
        docs = self._multi_search(queries, top_k_per_query=self.top_k)
        results, max_score = self._normalize_scores(docs)

        from core.config import RELEVANCE_THRESHOLD
        low_confidence = max_score < RELEVANCE_THRESHOLD

        output = {
            "results": results,
            "max_score": max_score,
            "low_confidence": low_confidence,
        }
        return json.dumps(output, ensure_ascii=False)


def create_rag(**kwargs) -> RAG:
    return RAG(**kwargs)
