# RAG 混合检索与上下文构建 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为当前朴素 RAG 增加混合检索（向量 + BM25 + RRF 重排）和结构化上下文构建。

**Architecture:** Search 路径变为三段管线 — RetrievalOptimizer 并行执行向量检索 + BM25 检索，RRF 融合排序后返回 top-k Document；ContextBuilder 将 Document 格式化为带元数据头的结构化字符串；RAG.search() 串联两者，初始化时从 ChromaDB 全量拉取 chunk 列表供 BM25 使用。

**Tech Stack:** langchain-chroma, langchain-community (BM25Retriever), rank_bm25, langchain-ollama (OllamaEmbeddings)

---

### Task 1: Install `rank_bm25` dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add rank_bm25 to requirements.txt**

```text
rank_bm25>=0.2.0
```

追加到 requirements.txt 末尾。

- [ ] **Step 2: Install the new dependency**

Run: `pip install rank_bm25>=0.2.0`
Expected: `Successfully installed rank_bm25-0.2.2` (or similar version)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add rank_bm25 dependency for hybrid retrieval"
```

---

### Task 2: Create `rag/retrieval.py` — RetrievalOptimizer

**Files:**
- Create: `rag/retrieval.py`

- [ ] **Step 1: Write the file**

```python
# rag/retrieval.py
"""混合检索优化器：向量检索 + BM25 关键词检索 + RRF 重排。"""

import logging
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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
            search_kwargs={"k": 5},
        )
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5,
        )
        logger.info("检索器初始化完成: vector(top_k=5) + BM25(top_k=5)")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """混合检索: 向量 + BM25 并行执行，RRF 融合后返回 top_k。"""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        reranked = self._rrf_rerank(vector_docs, bm25_docs, k=60)
        return reranked[:top_k]

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        """先混合检索扩大候选池 (top_k * 3)，再按元数据过滤。"""
        candidates = self.hybrid_search(query, top_k * 3)
        filtered = []
        for doc in candidates:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                else:
                    if doc.metadata[key] != value:
                        match = False
                        break
            if match:
                filtered.append(doc)
                if len(filtered) >= top_k:
                    break
        return filtered

    @staticmethod
    def _rrf_rerank(
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
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
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=. python3 -c "from rag.retrieval import RetrievalOptimizer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag/retrieval.py
git commit -m "feat: add RetrievalOptimizer with hybrid search and RRF rerank"
```

---

### Task 3: Create `rag/context.py` — ContextBuilder

**Files:**
- Create: `rag/context.py`

- [ ] **Step 1: Write the file**

```python
# rag/context.py
"""上下文构建器：将检索到的 Document 格式化为结构化文本。"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContextBuilder:
    """上下文构建器。

    将 Document 列表格式化为带元数据头的结构化字符串。

    用法:
        context = ContextBuilder.build_context(docs, max_length=2000)
    """

    @staticmethod
    def build_context(docs: List[Document], max_length: int = 2000) -> str:
        """将检索到的文档块格式化为结构化的上下文字符串。

        输出格式:
            【文档 1】 文件名 | 来源: path | 相关性: 0.0321
            文档内容...

            ==================================================
            【文档 2】 ...
        """
        if not docs:
            return "暂无相关知识。"

        parts = []
        length = 0
        separator = "\n" + "=" * 50 + "\n"

        for i, doc in enumerate(docs, 1):
            meta = f"【文档 {i}】"
            if "file_name" in doc.metadata:
                meta += f" {doc.metadata['file_name']}"
            if "source" in doc.metadata:
                meta += f" | 来源: {doc.metadata['source']}"
            if "rrf_score" in doc.metadata:
                meta += f" | 相关性: {doc.metadata['rrf_score']:.4f}"

            block = f"{meta}\n{doc.page_content}"

            if length + len(block) > max_length:
                break

            parts.append(block)
            length += len(block)

        return separator.join(parts)
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=. python3 -c "from rag.context import ContextBuilder; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag/context.py
git commit -m "feat: add ContextBuilder for structured document formatting"
```

---

### Task 4: Update `rag/knowledge_qa.py` — Integrate hybrid search and context building

**Files:**
- Modify: `rag/knowledge_qa.py` (entire file rewrite)

- [ ] **Step 1: Rewrite the file**

```python
# rag/knowledge_qa.py
"""RAG 问答引擎：混合检索（向量 + BM25 + RRF）+ 结构化上下文 + LLM 生成。"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from rag.retrieval import RetrievalOptimizer
from rag.context import ContextBuilder


class RAG:
    """RAG 引擎：混合检索 + LLM 生成。

    search(query) → str   返回结构化上下文（供 Agent 工具链使用）
    answer(query) → str   检索 + LLM 生成完整回答
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "lrs33/bce-embedding-base_v1:latest",
        llm_model: str = "qwen2.5:3b",
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
        )

        # 从 ChromaDB 全量拉取文档块，构建 BM25 索引（启动时一次性）
        print("构建混合检索器...")
        all_data = self.vector_store.get(include=["documents", "metadatas"])
        chunks = []
        for i in range(len(all_data.get("ids", []))):
            chunks.append(Document(
                page_content=all_data["documents"][i],
                metadata=all_data["metadatas"][i] if all_data.get("metadatas") else {},
            ))

        self.optimizer = RetrievalOptimizer(self.vector_store, chunks)

        print("初始化 LLM...")
        self.llm = ChatOllama(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0,
        )

        self.qa_prompt = ChatPromptTemplate.from_template("""你是一个专业的问答助手。请**仅基于**以下参考资料回答用户问题。
如果参考资料中没有相关信息，请如实告知用户，不要编造任何内容。

【参考资料】
{context}

【用户问题】
{question}

【回答】""")

    def search(self, query: str) -> str:
        """混合检索并返回格式化上下文字符串。"""
        docs = self.optimizer.hybrid_search(query, top_k=self.top_k)
        if not docs:
            return f"未找到与「{query}」相关的文档。"
        print(f"[RAG] 检索命中 {len(docs)} 个片段")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.metadata.get('file_name', '未知')} "
                  f"(rrf: {doc.metadata.get('rrf_score', 0):.4f})")
        return ContextBuilder.build_context(docs)

    def answer(self, question: str) -> str:
        """检索并生成完整回答。"""
        docs = self.optimizer.hybrid_search(question, top_k=self.top_k)
        if not docs:
            return "抱歉，知识库中没有找到相关信息。"

        context = ContextBuilder.build_context(docs)
        chain = self.qa_prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        print(f"[RAG] 检索命中 {len(docs)} 个相关片段")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.metadata.get('file_name', '未知')} "
                  f"(rrf: {doc.metadata.get('rrf_score', 0):.4f})")
        return response.content.strip()


def create_rag(**kwargs) -> RAG:
    return RAG(**kwargs)
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=. python3 -c "from rag.knowledge_qa import RAG; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag/knowledge_qa.py
git commit -m "refactor: integrate hybrid search and context builder into RAG"
```

---

### Task 5: Update `rag/__init__.py` — Add new exports

**Files:**
- Modify: `rag/__init__.py`

- [ ] **Step 1: Update exports**

Replace the file content with:

```python
# rag/__init__.py
from rag.knowledge_qa import RAG, create_rag
from rag.document_loader import load_documents_from_folder, split_documents
from rag.retrieval import RetrievalOptimizer
from rag.context import ContextBuilder
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=. python3 -c "from rag import RAG, RetrievalOptimizer, ContextBuilder; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag/__init__.py
git commit -m "feat: export RetrievalOptimizer and ContextBuilder from rag package"
```

---

### Task 6: Update `core/tools.py` — Simplify search_knowledge_base

**Files:**
- Modify: `core/tools.py` (lines 38-57)

- [ ] **Step 1: Edit search_knowledge_base function**

Replace the function body. Old:
```python
def search_knowledge_base(query: str) -> str:
    """从知识库中检索相关文档。

    依赖 RAG 的 search() 方法，返回原始文档片段（不含 LLM 生成），
    供业务 Agent 的 LLM 自行综合回答。
    """
    if _rag_client is None:
        return (
            "知识库尚未配置，无法检索相关文档。"
            "请联系管理员执行 configure_rag() 配置 RAG 检索器。"
        )
    try:
        results = _rag_client.search(query)
        if not results:
            return f"未找到与「{query}」相关的文档。"
        return "\n\n".join(results)
    except Exception as e:
        logger.error("[tools] RAG 检索失败: %s", e)
        return f"知识库检索异常：{e}"
```

New:
```python
def search_knowledge_base(query: str) -> str:
    """从知识库中检索相关文档。

    依赖 RAG 的 search() 方法，返回已格式化的结构化上下文（含元数据头），
    供业务 Agent 的 LLM 自行综合回答。
    """
    if _rag_client is None:
        return (
            "知识库尚未配置，无法检索相关文档。"
            "请联系管理员执行 configure_rag() 配置 RAG 检索器。"
        )
    try:
        result = _rag_client.search(query)
        if not result:
            return f"未找到与「{query}」相关的文档。"
        return result
    except Exception as e:
        logger.error("[tools] RAG 检索失败: %s", e)
        return f"知识库检索异常：{e}"
```

(The key change: `results` → `result`, no more `"\n\n".join()` since `search()` now returns a formatted string directly.)

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=. python3 -c "from core.tools import search_knowledge_base; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add core/tools.py
git commit -m "refactor: simplify search_knowledge_base to use new search() → str"
```

---

### Task 7: Integration test — Full pipeline verification

**Files:**
- No files created/modified (manual verification)

- [ ] **Step 1: Rebuild index to ensure clean state**

Run:
```bash
PYTHONPATH=. python3 rag/build_index.py --docs-dir ./docs --persist-dir ./chroma_db
```
Expected: Index builds successfully with chunks listed.

- [ ] **Step 2: Test RAG standalone**

Run:
```bash
PYTHONPATH=. python3 -c "
from rag import RAG

rag = RAG(persist_directory='./chroma_db', top_k=3)

# Test search() returns str (not list)
result = rag.search('退款政策')
assert isinstance(result, str), f'Expected str, got {type(result)}'
print('=== search() result ===')
print(result[:500])

# Test that result contains metadata header
assert '【文档' in result, 'Missing structured header'
assert '相关性' in result, 'Missing rrf_score in header'
print()
print('All assertions passed.')
"
```
Expected: Structured output with metadata headers and rrf scores. All assertions pass.

- [ ] **Step 3: Test answer() still works**

Run:
```bash
PYTHONPATH=. python3 -c "
from rag import RAG

rag = RAG(persist_directory='./chroma_db', top_k=3)
answer = rag.answer('退款相关政策是什么')
assert isinstance(answer, str) and len(answer) > 0
print('answer() OK:', answer[:200])
"
```
Expected: LLM-generated answer based on retrieved context.

- [ ] **Step 4: Test search_knowledge_base integration**

Run:
```bash
PYTHONPATH=. python3 -c "
from rag import RAG
from core.tools import configure_rag, search_knowledge_base

rag = RAG(persist_directory='./chroma_db', top_k=3)
configure_rag(rag)

result = search_knowledge_base('退款政策')
assert isinstance(result, str) and len(result) > 0
assert '【文档' in result
print('search_knowledge_base integration OK')
print(result[:300])
"
```
Expected: Formatted context string from `search()` passed through cleanly.

- [ ] **Step 5: Run existing test.py — no regressions**

Run:
```bash
python3 test.py
```
Expected: All 6 scenarios pass, exit code 0.

- [ ] **Step 6: Commit (if changes made)**

```bash
# Only if any fixes were applied during testing
git add -A && git commit -m "test: verify hybrid search and context builder integration"
```
