# Agentic RAG 设计文档

> 将当前固定检索升级为 Agent 自主决策检索：查询改写 → 多查询并行 → 重排归一化 → 置信度兜底 → JSON 返回。

**依赖：** Ollama (qwen2.5:3b 做查询改写), ChromaDB, rank_bm25, langchain-chroma

---

## 架构

```
query (原始用户问题)
  │
  ├─ 1. 检索次数检查 (conversation_id 维度，超 3 次返回拒绝 JSON)
  │
  ├─ 2. 查询改写 (qwen2.5:3b, 生成 1-3 个查询)
  │     原始 + 改写 → 候选查询列表
  │
  ├─ 3. 多查询并行检索
  │     每个查询 → hybrid_search(top_k=3) → 合并
  │
  ├─ 4. 去重 (按 page_content hash)
  │
  ├─ 5. 重排 & 归一化 (rrf_score / max_rrf_score → 0-1 区间)
  │
  ├─ 6. 置信度判断 (max_score < 0.5 → low_confidence=true)
  │
  └─ 7. 返回 JSON 字符串
```

## 返回 JSON 格式

```json
{
  "results": [
    {"text": "文档块内容1", "score": 0.92, "metadata": {"file_name": "...", "source": "..."}},
    {"text": "文档块内容2", "score": 0.78, "metadata": {...}}
  ],
  "max_score": 0.92,
  "low_confidence": false
}
```

检索次数超限或未配置 RAG 时：
```json
{"results": [], "max_score": 0, "low_confidence": true, "error": "检索次数已达上限(3次)"}
```

## 置信度规则

| 条件 | 行为 |
|------|------|
| `max_score >= 0.5` | `low_confidence: false`，Agent 正常引用 |
| `max_score < 0.5` | `low_confidence: true`，Agent 告知用户无可靠信息 |
| 连续 2 次 `low_confidence: true` | 后续不再检索，直接答不知道 |

## 查询改写

模型：Ollama `qwen2.5:3b`（与 embedding 分开）

Prompt：
```
你是一个查询改写专家。根据对话历史，将用户问题改写为1-3个适合向量检索的独立查询。
每行一个查询，不要序号，不要额外解释。
如果原问题已经清晰明确，直接原样返回即可。

对话历史：{history}
用户问题：{query}
```

解析：按行 split，空行过滤，与原问题去重。

## 多查询检索 + 去重

每个查询 `hybrid_search(query, top_k=3)` → 合并 → `hash(page_content)` 去重（保留更高 RRF 分数的那条）。

## 重排 & 归一化

取所有去重结果的最大 RRF 分数为分母：`score = rrf_score / max_rrf_score`，使得分数落在 0-1 区间。

---

## 文件变更

### 修改：`rag/knowledge_qa.py`

`RAG` 重写：

- `__init__` 新增参数：`rewrite_model`（查询改写 LLM 模型名）
- 新增 `_rewrite_query(query, history)` — 调用 qwen2.5:3b 改写，返回 `list[str]`
- 新增 `_multi_search(queries, top_k_per_query)` — 并行检索、合并去重
- 新增 `_normalize_scores(docs)` — RRF 分数归一化到 0-1
- `search(query, history="")` → 走完整管线，返回 JSON `str`

移除 `ContextBuilder` 调用（返回值从格式化文本变为 JSON）。

### 修改：`core/tools.py`

- `search_knowledge_base` 新增 `conversation_id` 参数
- `configure_rag` 接受 `RAG`，不变
- 维护 `_retrieval_counts: dict[str, int]`，按 conversation_id 计数
- 超限返回拒绝 JSON

### 修改：`core/config.py`

新增：
```python
RELEVANCE_THRESHOLD = 0.5
MAX_RETRIEVAL_ROUNDS = 3
REWRITE_LLM_CONFIG = {
    "provider": "ollama",
    "model": "qwen2.5:3b",
    "api_base": "http://127.0.0.1:11434",
    "temperature": 0.1,
}
```

### 修改：`core/prompts.py`

业务 Agent prompt 追加：
```
你拥有 search_knowledge_base 工具。调用后你会得到一个 JSON：
- 如果 low_confidence 为 true（max_score < 0.5），不要编造，告知用户"没有找到足够相关的信息"。
- 如果 low_confidence 为 false，使用 results 中的 text 生成回答，优先引用分数高的。
```

### 不变

- `rag/retrieval.py` — `RetrievalOptimizer` 不变
- `rag/context.py` — `ContextBuilder` 保留但不再被 `search()` 调用
- `rag/build_index.py` — 不变

---

## 测试要点

1. 查询改写生成 1-3 个查询，不丢失原意
2. 多查询检索去重正确
3. 分数归一化后 max_score 在 0-1 区间
4. `low_confidence` 在阈值下为 true
5. 同一 conversation_id 第 4 次调用返回拒绝 JSON
6. `test.py` 无回归
