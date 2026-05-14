# RAG 混合检索与上下文构建 设计文档

> 基于 `docs/rag_retrieval_generation_reference.py`，为当前朴素 RAG 增加混合检索（向量 + BM25 + RRF 重排）和结构化上下文构建。

**范围外：** 查询重写不在此次范围。

---

## 架构

```
query (用户问题)
  │
  ▼
┌──────────────────────────────────────┐
│  RetrievalOptimizer                   │
│  ├─ 向量检索 (ChromaDB, top_k=5)      │
│  ├─ BM25 关键词检索 (top_k=5)         │
│  └─ RRF (Reciprocal Rank Fusion) 排序 │
│      score = Σ 1/(k + rank)           │
└──────────────────────────────────────┘
  │  List[Document] (metadata 含 rrf_score)
  ▼
┌──────────────────────────────────────┐
│  ContextBuilder.build_context()       │
│  ├─ 元数据头: 文件名 | 来源 | 相关性   │
│  ├─ 结构化拼接 (max_length 截断)      │
│  └─ 返回格式化字符串                   │
└──────────────────────────────────────┘
  │  str
  ▼
返回给 search_knowledge_base → Agent
```

---

## 文件变更

### 新建：`rag/retrieval.py`

`RetrievalOptimizer` 类：
- `__init__(vectorstore, chunks)`：从 ChromaDB 初始化向量检索器 + 从 chunks 构建 BM25 检索器
- `hybrid_search(query, top_k=3)`：并行执行向量 + BM25 → RRF 融合 → 返回 top_k
- `_rrf_rerank(vector_docs, bm25_docs, k=60)`：静态方法，RRF 融合算法

依赖：`langchain_chroma.Chroma`, `langchain_community.retrievers.BM25Retriever`, `rank_bm25`

### 新建：`rag/context.py`

`ContextBuilder` 类：
- `build_context(docs, max_length=2000)`：静态方法，将 Document 列表格式化为：

```
【文档 1】 文件名 | 来源: /path/to/file | 相关性: 0.0321
文档内容...

==================================================
【文档 2】 文件名 | 来源: /path/to/file | 相关性: 0.0298
文档内容...
```

若无结果返回 `"暂无相关知识。"`。

### 修改：`rag/knowledge_qa.py`

`RAG` 内部重构：

```python
class RAG:
    def __init__(self, ...):
        # 连接 ChromaDB，获取 embeddings + vector_store
        self.embeddings = OllamaEmbeddings(...)
        self.vector_store = Chroma(...)

        # 从 ChromaDB 全量拉取文档块（启动时一次性）
        all_docs = self.vector_store.get(include=["documents", "metadatas"])
        chunks = [Document(page_content=... , metadata=...) for ...]

        # 构建混合检索器
        self.optimizer = RetrievalOptimizer(self.vector_store, chunks)

    def search(self, query: str) -> str:
        docs = self.optimizer.hybrid_search(query, top_k=self.top_k)
        return ContextBuilder.build_context(docs, max_length=2000)
```

- `search()` 返回 `str`（已格式化的上下文）而非 `list[str]`
- `answer()` 保持不变，内部仍用向量检索 + LLM 生成
- `top_k` 控制混合检索最终返回数（默认 3）

### 修改：`core/tools.py`

`search_knowledge_base` 简化：`RAG.search()` 现在直接返回格式化字符串，不需要再手动 `"\n\n".join(results)`。

```python
def search_knowledge_base(query: str) -> str:
    if _rag_client is None:
        return "知识库尚未配置..."
    try:
        result = _rag_client.search(query)
        if not result:
            return f"未找到与「{query}」相关的文档。"
        return result
    except Exception as e:
        ...
```

### 修改：`rag/build_index.py`

不变，仍使用 `from langchain_chroma import Chroma`。

### 修改：`rag/__init__.py`

```python
from rag.knowledge_qa import RAG, create_rag
from rag.document_loader import load_documents_from_folder, split_documents
from rag.retrieval import RetrievalOptimizer      # 新增
from rag.context import ContextBuilder             # 新增
```

### 修改：`requirements.txt`

新增 `rank_bm25>=0.2.0`。

---

## 接口兼容性

| 接口 | 变更前 | 变更后 |
|------|--------|--------|
| `RAG.search(query)` | 返回 `list[str]` | 返回 `str`（格式化上下文） |
| `RAG.answer(query)` | 返回 `str` | 不变 |
| `configure_rag(client)` | 接受 `RAG` | 不变 |
| `search_knowledge_base(query)` | 返回 `str` | 返回值不变，内部逻辑简化 |
| `build_index` | CLI 脚本 | 不变 |

**breaking change:** `RAG.search()` 返回类型从 `list[str]` 变为 `str`。当前只有 `search_knowledge_base` 调用 `search()`，已一并适配。

---

## 测试要点

1. `RetrievalOptimizer.hybrid_search` 返回的 Document 包含 `rrf_score` 元数据
2. `ContextBuilder.build_context` 格式化输出包含文件名、来源、相关性分数
3. `RAG.search()` 返回格式化字符串（非列表）
4. `build_index` 无回归
5. 无文档时 `search()` 返回 `"未找到..."` 提示
6. 现有 `test.py` 全部通过（RAG 工具路径未变）
