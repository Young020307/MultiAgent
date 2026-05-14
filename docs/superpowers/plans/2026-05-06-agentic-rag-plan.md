# Agentic RAG 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前固定检索升级为 Agent 自主决策检索：查询改写 → 多查询并行 → 重排归一化 → 置信度兜底 → JSON 返回。

**Architecture:** `search_knowledge_base(conversation_id)` 内部自动走完整管线：次数检查 → 查询改写 (qwen2.5:3b) → 多查询并行检索 → hash 去重 → RRF 分数归一化到 0-1 → 置信度判断 → JSON 返回。Agent prompt 增加 JSON 解析规则，根据 `low_confidence` 决定是否引用检索结果。

**Tech Stack:** Ollama (qwen2.5:3b 查询改写), ChromaDB, rank_bm25, langchain-chroma, langchain-ollama

---

### Task 1: Add Agentic RAG config to `core/config.py`

**Files:**
- Modify: `core/config.py`

- [ ] **Step 1: Append config constants**

```python
# ================================================================
# Agentic RAG 配置
# ================================================================

RELEVANCE_THRESHOLD = 0.5
MAX_RETRIEVAL_ROUNDS = 3
REWRITE_LLM_CONFIG = {
    "provider": "ollama",
    "model": "qwen2.5:3b",
    "api_base": "http://127.0.0.1:11434",
    "temperature": 0.1,
}
```

位置：追加到 `core/config.py` 文件末尾。

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.config import RELEVANCE_THRESHOLD, MAX_RETRIEVAL_ROUNDS, REWRITE_LLM_CONFIG; print(RELEVANCE_THRESHOLD, MAX_RETRIEVAL_ROUNDS, REWRITE_LLM_CONFIG['model'])"`
Expected: `0.5 3 qwen2.5:3b`

- [ ] **Step 3: Skip commit (no git)**

---

### Task 2: Rewrite `rag/knowledge_qa.py` — add query rewrite, multi-search, normalize, JSON output

**Files:**
- Modify: `rag/knowledge_qa.py` (entire file)

- [ ] **Step 1: Write the complete new file**

```python
# rag/knowledge_qa.py
"""Agentic RAG 引擎：查询改写 → 多查询并行 → 重排归一化 → JSON 返回。"""

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

        # 去重，保留原始查询在最前
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
        seen: dict[int, Document] = {}
        for q in queries:
            docs = self.optimizer.hybrid_search(q, top_k=top_k_per_query)
            for doc in docs:
                doc_id = hash(doc.page_content)
                if doc_id not in seen:
                    seen[doc_id] = doc
                else:
                    # 保留 RRF 分数更高的
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
        # 1. 查询改写
        queries = self._rewrite_query(query, history)

        # 2. 多查询检索 + 去重
        docs = self._multi_search(queries, top_k_per_query=self.top_k)

        # 3. 归一化
        results, max_score = self._normalize_scores(docs)

        # 4. 置信度判断
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
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from rag.knowledge_qa import RAG; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Test standalone search with JSON output**

Run:
```bash
PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "
from rag import RAG
rag = RAG(persist_directory='./chroma_db', top_k=3)
result = rag.search('退换货政策')
import json
data = json.loads(result)
assert 'results' in data
assert 'max_score' in data
assert 'low_confidence' in data
assert isinstance(data['results'], list)
for r in data['results']:
    assert 'text' in r
    assert 'score' in r
    assert 0 <= r['score'] <= 1
print('max_score:', data['max_score'])
print('low_confidence:', data['low_confidence'])
print('result count:', len(data['results']))
print('OK')
"
```
Expected: Valid JSON with correct structure, scores in 0-1 range.

- [ ] **Step 4: Skip commit (no git)**

---

### Task 3: Update `core/tools.py` — retrieval counting + JSON return

**Files:**
- Modify: `core/tools.py`

- [ ] **Step 1: Edit the RAG section of tools.py**

Replace the RAG section (lines 18-57) with:

```python
# RAG 客户端 + 检索计数
_rag_client = None
_retrieval_counts: dict[str, int] = {}
_consecutive_low_conf: dict[str, int] = {}

from core.config import MAX_RETRIEVAL_ROUNDS


def configure_rag(client):
    """配置 RAG 客户端。

    传入 rag.RAG 实例。
    示例：
        from rag import RAG
        configure_rag(RAG(persist_directory="./chroma_db"))
    """
    global _rag_client
    _rag_client = client
    logger.info("[tools] RAG 客户端已配置")


def search_knowledge_base(query: str, conversation_id: str = "") -> str:
    """Agentic RAG 检索工具。

    内部自动完成查询改写、多查询并行、去重、归一化、置信度判断，
    返回 JSON 字符串。

    Args:
        query: 检索查询词
        conversation_id: 对话 ID，用于检索次数限制和低置信度追踪

    Returns:
        JSON 字符串，结构：
        {"results": [{"text": ..., "score": ..., "metadata": {...}}],
         "max_score": ..., "low_confidence": ...}
    """
    import json

    if _rag_client is None:
        return json.dumps({
            "results": [],
            "max_score": 0,
            "low_confidence": True,
            "error": "知识库尚未配置，无法检索。",
        }, ensure_ascii=False)

    # 检索次数检查
    if conversation_id:
        count = _retrieval_counts.get(conversation_id, 0)
        # 连续低置信度短路
        if _consecutive_low_conf.get(conversation_id, 0) >= 2:
            return json.dumps({
                "results": [],
                "max_score": 0,
                "low_confidence": True,
                "error": "前两次检索均为低置信度，已停止继续检索。",
            }, ensure_ascii=False)
        if count >= MAX_RETRIEVAL_ROUNDS:
            return json.dumps({
                "results": [],
                "max_score": 0,
                "low_confidence": True,
                "error": f"检索次数已达上限({MAX_RETRIEVAL_ROUNDS}次)。",
            }, ensure_ascii=False)

    try:
        result_json = _rag_client.search(query)

        # 追踪检索次数和低置信度
        if conversation_id:
            _retrieval_counts[conversation_id] = _retrieval_counts.get(conversation_id, 0) + 1
            data = json.loads(result_json)
            if data.get("low_confidence"):
                _consecutive_low_conf[conversation_id] = _consecutive_low_conf.get(conversation_id, 0) + 1
            else:
                _consecutive_low_conf[conversation_id] = 0

        return result_json
    except Exception as e:
        logger.error("[tools] RAG 检索失败: %s", e)
        return json.dumps({
            "results": [],
            "max_score": 0,
            "low_confidence": True,
            "error": f"知识库检索异常：{e}",
        }, ensure_ascii=False)
```

需要同时修改 `import json` 放在文件顶部（如果还没有的话），以及确保 `from core.config import MAX_RETRIEVAL_ROUNDS` 在 RAG 区域。

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.tools import search_knowledge_base, configure_rag; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Test retrieval count enforcement**

Run:
```bash
PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "
from rag import RAG
from core.tools import configure_rag, search_knowledge_base
import json, core.tools as t

rag = RAG(persist_directory='./chroma_db', top_k=3)
configure_rag(rag)

cid = 'test-counting'
# Call 1
r1 = json.loads(search_knowledge_base('退款', cid))
print('Call 1 - count:', t._retrieval_counts.get(cid), 'low_conf:', r1['low_confidence'])
# Call 2
r2 = json.loads(search_knowledge_base('价格', cid))
print('Call 2 - count:', t._retrieval_counts.get(cid))
# Call 3
r3 = json.loads(search_knowledge_base('订单', cid))
print('Call 3 - count:', t._retrieval_counts.get(cid))
# Call 4 — should hit limit
r4 = json.loads(search_knowledge_base('售后', cid))
print('Call 4 - count:', t._retrieval_counts.get(cid))
assert '检索次数已达上限' in r4.get('error', '')
print('OK - retrieval limit enforced')
# Cleanup
t._retrieval_counts.pop(cid, None)
t._consecutive_low_conf.pop(cid, None)
"
```
Expected: Call 4 returns error JSON with "检索次数已达上限".

- [ ] **Step 4: Skip commit (no git)**

---

### Task 4: Update `core/prompts.py` — add JSON parsing rule to agent prompts

**Files:**
- Modify: `core/prompts.py`

- [ ] **Step 1: Append JSON usage rule to technical, sales, support prompts**

For each of `"technical"`, `"sales"`, `"support"` prompts, append the following rule before the closing `"""`:

```
你拥有 search_knowledge_base 工具。调用后你会得到一个 JSON：
- 如果 low_confidence 为 true，不要编造，告知用户"没有找到足够相关的信息，建议联系人工客服"。
- 如果 low_confidence 为 false，使用 results 中的 text 生成回答，优先引用分数(score)高的结果。
```

The edit — for each prompt ending with `请提供专业、详细的技术解答。"""` or similar, change to include the rule.

具体修改：

`"technical"` prompt 末尾 `"""` 前追加：
```

你拥有 search_knowledge_base 工具。调用后你会得到一个 JSON：
- 如果 low_confidence 为 true，不要编造，告知用户"没有找到足够相关的信息，建议联系人工客服"。
- 如果 low_confidence 为 false，使用 results 中的 text 生成回答，优先引用分数(score)高的结果。"""```

`"sales"` prompt 末尾 `"""` 前追加同样内容。

`"support"` prompt 末尾 `"""` 前追加同样内容。

- [ ] **Step 2: Verify prompts parse correctly**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.prompts import PROMPTS; print('OK')" 2>/dev/null`
Expected: `OK`

- [ ] **Step 3: Skip commit (no git)**

---

### Task 5: Update `test/test_rag.py` — JSON verification tests

**Files:**
- Modify: `test/test_rag.py`

- [ ] **Step 1: Rewrite test file**

```python
# test/test_rag.py
"""Agentic RAG 检索测试 — 单问题检索，验证 JSON 输出结构。"""

import json
import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

from rag import RAG

TEST_QUESTION = "什么情况下能退款"


def main():
    rag = RAG(persist_directory="./chroma_db", top_k=3)
    result = rag.search(TEST_QUESTION)
    data = json.loads(result)

    print(f"问题：{TEST_QUESTION}")
    print(f"结果数：{len(data['results'])}")
    print(f"max_score：{data['max_score']:.4f}")
    print(f"low_confidence：{data['low_confidence']}")
    print("=" * 60)
    for i, r in enumerate(data["results"], 1):
        print(f"\n--- 结果 {i} (score: {r['score']:.4f}) ---")
        print(f"来源：{r['metadata'].get('file_name', '?')}")
        print(f"内容({len(r['text'])}字)：{r['text'][:300]}...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 test/test_rag.py`
Expected: JSON output with scores, max_score, low_confidence. Results contain file_name in metadata.

- [ ] **Step 3: Skip commit (no git)**

---

### Task 6: Integration test — run `test/test.py` for regression

**Files:**
- No changes (verification only)

- [ ] **Step 1: Run full test.py**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 test/test.py`
Expected: All 6 scenarios pass, exit code 0.

Note: Business agents (technical, sales, support) may now call `search_knowledge_base` and receive JSON — their LLM must parse and handle the JSON response. The test.py scenarios that trigger knowledge search should still produce reasonable answers.

- [ ] **Step 2: Skip commit (no git)**
