# 智能客服多智能体系统 — 架构设计文档

> 2026-05-07 | LangGraph 1.x + ChromaDB + Ollama + MiniMax

---

## 1. 系统概述

基于 LangGraph 构建的多智能体智能客服系统。核心思路：一个 Supervisor Router 统一意图分类和安检，下游四个业务 Agent 各司其职，全部通过共享 State 通信，单向流无回环。

**LLM 调度：** Router 用本地 Ollama (qwen2.5:7b) 零成本分类，业务 Agent 用 MiniMax API (M2.5-highspeed) 保证回复质量。

**RAG：** Agentic RAG — 查询改写 → 多查询并行混合检索（向量 + BM25）→ RRF融合得分 → 重排 → JSON 返回，Agent 自主决定是否检索。

**人工介入：** 三级触发（用户请求 / 系统故障 / 负面情感），LangGraph interrupt 真挂起，人工坐席可多轮沟通后 resume。

---

## 2. 系统架构图

```
                            ┌─────────────────────┐
                            │   supervisor_router  │
                            │                      │
                            │  ├─ LLM 意图分类      │
                            │  ├─ 关键词情感检测    │
                            │  └─ 异常计数器检查    │
                            └──────┬───────────────┘
                                   │
                    ┌──────────────┼──────────────────┐
                    │              │                   │
                    ▼              ▼                   ▼
          ┌──────────┐   ┌──────────────┐    ┌──────────────┐
          │ escalate │   │ greeting /   │    │ technical    │
          │ sentiment│   │ ambiguous    │    │ sales        │
          │ anomaly  │   │              │    │ support      │
          │          │   │ conversation │    │ feedback     │
          │ human_   │   │ _agent       │    │ (业务Agent)  │
          │ handoff  │   └──────┬───────┘    └──────┬───────┘
          │ (interrupt!)        │                   │
          └────┬────┘           │         ┌─────────┴─────────┐
               │                │         ▼                   ▼
               │                │   ┌──────────┐     ┌─────────────┐
               ▼                │   │ quality  │     │  fallback   │
              END               │   │ _check   │     │  _reply     │
                                │   └────┬─────┘     └──────┬──────┘
                                │        │                  │
                                │   ┌────┴────┐             │
                                │   ▼         ▼             │
                                │ human  auto_              │
                                │_review  approve           │
                                │   │         │             │
                                │   └────┬────┘             │
                                │        ▼                  │
                                │  ┌──────────┐             │
                                └─▶│ format   │◀────────────┘
                                   │ _response│
                                   └────┬─────┘
                                        ▼
                                       END
```

---

## 3. 核心组件

### 3.1 State — 共享状态 (`core/state.py`)

整个图唯一的数据载体，TypedDict 定义，`messages` 字段累加（`operator.add`），其余字段覆盖更新。

```python
class AgentState(TypedDict):
    conversation_id: str                              # 对话 ID (UUID 前8位)
    messages: Annotated[List[Message], operator.add]  # 消息累积
    intent: IntentType                                # 7 种意图之一
    routing_target: str                               # 下一跳节点名
    agent_response: str                              # 业务 Agent 原始回复
    quality_score: int                                # 质检评分 1-5
    quality_reason: str                               # 评分理由
    requires_human_review: bool                       # score < 3 触发
    final_response: str                               # 格式化后的最终回复
    conversation_log: list[LogEntry]                  # 历史轮次摘要
    escalated: bool                                   # 是否已触发人工交接
    escalate_reason: str                              # user_requested / system_failure / negative_sentiment
    handoff_context: dict                             # 交接给人工坐席的上下文包
    anomaly_counters: dict                            # {low_conf_streak, quality_fail_streak,...}
    sentiment: str                                    # neutral / negative / critical
    metadata: dict                                    # 自由扩展
```

**结构化输出模型（Pydantic）：**

| 模型 | 字段 | 用途 |
|------|------|------|
| `RouterDecision` | `intent`, `confidence` | Router LLM 返回结构 |
| `QualityOutput` | `score` (1-5), `reason` | 质检 LLM 返回结构 |

**意图类型（7 种）：**

| 意图 | 路由目标 |
|------|----------|
| `greeting` / `ambiguous` | `conversation_agent` |
| `technical` | `technical` Agent |
| `sales` | `sales` Agent |
| `support` | `support` Agent |
| `feedback` | `feedback` Agent |
| `escalate` | `human_handoff` (直接人工) |

### 3.2 Graph — 图流程 (`core/graph.py`)

**单条正向流，无回环。** 关键路由逻辑：

```
入口: supervisor_router
  │
  ├─ routing_target == "human_handoff" ──→ human_handoff (interrupt!) → END
  │
  ├─ routing_target in {technical, sales, support, feedback}
  │    └─→ 业务 Agent
  │         ├─ agent_response 非空 → quality_check
  │         │    ├─ score >= 3 → auto_approve → format_response
  │         │    └─ score < 3  → human_review（加 [已审核] 标签）→ format_response
  │         └─ agent_response 为空 → fallback → format_response
  │
  └─ 其他 → conversation_agent → format_response → END
```

**LangGraph interrupt：** 编译时设置 `interrupt_before=["human_handoff"]`，图进入该节点前挂起，等人工 `update_state` 后 `invoke(None)` resume。

### 3.3 Agent 工厂 (`core/agent_factory.py`)

每个 Agent 独立 LangChain `create_agent` 实例 + 独立 `MemorySaver`。

**LLM 懒加载：** `get_llm_for(name)` 按 provider 分派：

| Agent | Provider | Model | Temperature |
|-------|----------|-------|-------------|
| `supervisor_router` | Ollama | qwen2.5:7b | 0.1 |
| `conversation_agent` | MiniMax (OpenAI compat) | M2.5-highspeed | 0.8 |
| `technical` | MiniMax | M2.5-highspeed | 0.3 |
| `sales` | MiniMax | M2.5-highspeed | 0.5 |
| `support` | MiniMax | M2.5-highspeed | 0.5 |
| `feedback` | MiniMax | M2.5-highspeed | 0.5 |
| `quality_check` | MiniMax | M2.5-highspeed | 0.1 |

**Memory 隔离：** 每个 Agent 的 checkpointer key 为 `{conv_id}_{agent_name}`，不同对话的 Agent 记忆互不污染。

**节点包装：** `wrap_agent_node(name)` 返回标准 `(state) → dict` 函数。业务 Agent 返回 `agent_response`，conversation_agent 直接返回 `final_response`。

### 3.4 RAG 系统 (`rag/`)

**架构管线：**

```
用户问题
  │
  ├─ 1. 查询改写 (qwen2.5:3b, 生成 1-3 个查询)
  ├─ 2. 多查询并行检索 (每个查询 → 向量 + BM25)
  ├─ 3. hash 去重
  ├─ 4. RRF 分数归一化到 0-1
  ├─ 5. 重排序 
  ├─ 6. 置信度判断 (max_score < 0.5 → low_confidence)
  └─ 7. JSON 返回
```

**组件：**

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `rag/document_loader.py` | `load_documents_from_folder`, `split_documents` | 文档加载 + 自适应分块 (.txt/.md/.pdf/.docx) |
| `rag/retrieval.py` | `RetrievalOptimizer` | 混合检索：向量相似度 (ChromaDB) + BM25 关键词 (rank_bm25)，RRF 融合，重排序 |
| `rag/context.py` | `ContextBuilder.build_context()` | Document → 结构化文本（当前 search 管线未使用，保留） |
| `rag/knowledge_qa.py` | `RAG` | 主引擎：查询改写 + 多查询检索 + 归一化 + JSON 输出 |
| `rag/build_index.py` | `rebuild_index()` | CLI：加载文档 → ChromaDB 持久化，固定 collection name |

**检索参数：**

| 参数 | 值 |
|------|----|
| 嵌入模型 | Ollama `lrs33/bce-embedding-base_v1:latest` |
| 查询改写模型 | Ollama `qwen2.5:3b` |
| 向量检索 top_k | 5 (候选池) |
| BM25 检索 top_k | 5 (候选池) |
| 混合检索最终 top_k | 3 (默认) |
| RRF 平滑参数 k | 60 |
| 低置信度阈值 | 0.5 |
| 最大检索次数 | 3 次/对话 |

**索引构建：**

```bash
PYTHONPATH=. python rag/build_index.py
  --docs-dir ./knowledge          # 知识文档目录
  --persist-dir ./chroma_db       # 向量库持久化目录
```

构建前自动删除旧索引 (`shutil.rmtree`)，collection name 固定为 `customer_service_knowledge`。

### 3.5 人工介入 (`core/node.py:human_handoff`)

**三大触发场景：**

| 场景 | 触发条件 | `escalate_reason` |
|------|---------|-------------------|
| 1. 用户要求转人工 | Router 识别 `escalate` 意图 | `user_requested` |
| 4. 系统故障兜底 | 异常计数器超阈值 | `system_failure` |
| 5. 负面情感升级 | 关键词命中 `CRITICAL_PATTERNS` | `negative_sentiment` |

**异常计数器阈值：**

| 计数器 | 触发条件 | 阈值 |
|--------|---------|------|
| `low_conf_streak` | 连续 RAG 低置信度 | ≥ 2 |
| `quality_fail_streak` | 质检失败 (score < 3) | ≥ 1 |
| `tool_error_streak` | 连续工具调用异常 | ≥ 3 |

成功回复后重置，失败累积。计数器在 `format_response` 中更新，下一轮 `supervisor_router` 检查。

**Interrupt 流程：**

```
supervisor_router → human_handoff [interrupt!]
                         │
      ══════ 人工坐席与用户多轮沟通 ══════
                         │
      graph.update_state(注入人工对话到 messages + conversation_log)
      graph.invoke(None) → human_handoff 节点执行 → END
```

注入后下一轮 supervisor 通过 `conversation_log` 可见人工交互上下文。

### 3.6 情感检测 (`core/node.py:_detect_sentiment`)

纯关键词匹配，零延迟，不调 LLM：

- **Critical (9 词)：** `投诉`, `举报`, `忍无可忍`, `我要告`, `再也不用了`, `骗子`, `垃圾公司`, `太失望了`, `糟透了`
- **Negative (9 词)：** `不满意`, `太差了`, `很生气`, `火大`, `什么玩意`, `糊弄`, `忽悠`, `扯淡`, `坑人`

### 3.7 数据库 (`core/db.py`)

SQLAlchemy ORM，默认 SQLite。5 张表：

| 表名 | 用途 | Agent 工具 |
|------|------|-----------|
| `pricing_plans` | 套餐价格 | `db_query_pricing` (sales) |
| `promotions` | 优惠码 | `db_check_promotion` (sales) |
| `orders` | 订单 | `db_lookup_order` (support) |
| `refunds` | 退款单 | `db_check_refund_status` (support) |
| `customer_feedback` | 用户反馈 | `db_log_feedback` (feedback) |

### 3.8 工具注册表 (`core/tools.py`)

```python
KNOWLEDGE_TOOLS = [search_knowledge_base]

BUSINESS_TOOLS = {
    "technical": [],
    "sales":     [db_query_pricing, db_check_promotion],
    "support":   [db_lookup_order, db_check_refund_status],
    "feedback":  [db_log_feedback],
}

# 通过 AGENT_TOOLS 组合
AGENT_TOOLS = {
    "technical": ["search_knowledge_base"],
    "sales":     ["db_query_pricing", "db_check_promotion", "search_knowledge_base"],
    "support":   ["db_lookup_order", "db_check_refund_status", "search_knowledge_base"],
    "feedback":  ["db_log_feedback"],
}
```

### 3.9 多轮对话上下文 (`conversation_log`)

每轮 `format_response` 追加一条 `LogEntry`：`{turn, agent, user, reply}`。纯自然语言摘要，不含工具调用/JSON。Router prompt 拼接历史轮次作为上下文，使 LLM 能跨轮理解对话主题切换。

---

## 4. 请求生命周期

```
1. 用户发送消息
     │
2. chat.py 构建 AgentState，分配 conversation_id
     │
3. graph.invoke(state) 进入 supervisor_router
     │
4. Router:
     ├─ 拼接 conversation_log 为上下文
     ├─ LLM 分类意图 (RouterDecision JSON)
     ├─ 关键词情感检测
     └─ 异常计数器检查
     │
5. route_after_router:
     ├─ escalate / sentiment=critical / anomaly → human_handoff (interrupt!)
     ├─ greeting / ambiguous → conversation_agent
     └─ business → 对应业务 Agent
     │
6. 业务 Agent:
     ├─ LangChain create_agent 决策
     ├─ 如需检索 → 调用 search_knowledge_base (Agentic RAG)
     │    └─ 内部：查询改写 → 多查询检索 → 去重归一化 → JSON 返回
     ├─ 如查数据库 → 调用 db_* 工具
     └─ 生成 agent_response
     │
7. quality_check: LLM 评分 1-5
     │
8. auto_approve / human_review: 组装 final_response
     │
9. format_response:
     ├─ 格式化签名块 (客服团队 + 工单号)
     ├─ 追加 conversation_log
     └─ 更新 anomaly_counters
     │
10. 返回 final_response 给用户
```

---

## 5. 关键设计决策

| 决策 | 理由 |
|------|------|
| 单向流，无回环 | 简单可控，避免无限循环；需要回环时靠 conversation_log 跨越轮次 |
| Router 用本地 Ollama 小模型 | 意图分类对准确度要求低，本地零成本延时 |
| 业务 Agent 用 MiniMax | 直接面向用户，回复质量重要 |
| 每 Agent 独立 MemorySaver | 避免跨 Agent 记忆污染 |
| RRF 而非 CrossEncoder 重排 | RRF 零延迟、零依赖；CrossEncoder 待后续加入 |
| 查询改写用 qwen2.5:3b | 与 embedding 模型分离，3b 足够 |
| 人工交接用 interrupt_before | 真正的 LangGraph 中断，人工可多轮沟通后 resume |
| 情感检测用关键词匹配 | 零延迟，误判率低（触发后有人工兜底） |
| conversation_log 纯自然语言 | 对 Router LLM 友好，不含噪声 |

---

## 6. 项目文件结构

```
MultiAgent/
├── core/                          # 核心模块
│   ├── state.py                   # AgentState, IntentType, Pydantic 模型
│   ├── graph.py                   # LangGraph 图构建和编译
│   ├── node.py                    # 所有节点函数 (router, handoff, quality, format)
│   ├── agent_factory.py           # LLM 工厂 + Agent 构建 + 节点包装
│   ├── prompts.py                 # 所有提示词模板
│   ├── config.py                  # 配置常量 (LLM, 阈值, 情感词)
│   ├── tools.py                   # RAG 工具 + 业务数据库工具
│   ├── db.py                      # SQLAlchemy ORM + 会话管理
│   └── logger.py                  # 日志工具
│
├── rag/                           # RAG 模块
│   ├── knowledge_qa.py            # RAG 主引擎 (查询改写 + 检索 + JSON)
│   ├── retrieval.py               # RetrievalOptimizer (向量 + BM25 + RRF)
│   ├── context.py                 # ContextBuilder (结构化格式化)
│   ├── document_loader.py         # 文档加载与自适应分块
│   ├── build_index.py             # CLI 索引构建脚本
│   └── __init__.py                # 包导出
│
├── knowledge/                     # 知识文档目录
├── chroma_db/                     # ChromaDB 持久化目录
├── customer_service.db            # SQLite 业务数据库
├── .env                           # 环境变量 (MINIMAX_API_KEY 等)
├── requirements.txt               # Python 依赖
│
├── test/                          # 测试
│   ├── test.py                    # 6 场景集成测试
│   └── test_rag.py                # RAG 检索测试
│
├── docs/                          # 文档
│   ├── ARCHITECTURE.md            # 本文件
│   ├── simple-rag-guide.md        # 朴素 RAG 指南
│   ├── rag_retrieval_generation_reference.py  # 参考实现
│   └── superpowers/
│       ├── specs/                 # 设计文档
│       └── plans/                 # 实现计划
│
└── CLAUDE.md                      # 项目行为规范
```

---

## 7. 依赖

```
langgraph>=0.2.0          # 图编排
langchain>=0.3            # Agent 框架
langchain-openai>=0.3     # MiniMax (OpenAI 兼容)
langchain-ollama>=0.2     # Ollama 本地模型
langchain-chroma>=1.0     # ChromaDB (新版)
langchain-community>=0.3  # BM25Retriever, 文档加载器
chromadb>=0.5             # 向量数据库
rank_bm25>=0.2            # BM25 关键词检索
pypdf>=4.0                # PDF 解析
docx2txt>=0.8             # DOCX 解析
unstructured>=0.15        # DOC 解析
pydantic>=2.0             # 数据模型
sqlalchemy>=2.0           # ORM
```

**外部服务：**
- Ollama (`127.0.0.1:11434`): embedding + Router + 查询改写
- MiniMax API: 业务 Agent + 质检
