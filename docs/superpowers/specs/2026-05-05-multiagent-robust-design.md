# MultiAgent 客户服务系统 — 健壮性增强设计

**日期**: 2026-05-05
**目标形态**: CLI 工具
**协调模式**: Supervisor（多指派 + 并行执行）

---

## 1. 文件结构

```
MultiAgent/
├── core/
│   ├── state.py        # 状态定义（Pydantic 结构化输出模型 + AgentState）
│   ├── prompts.py      # [新] 集中管理所有提示词模板
│   ├── node.py          # 节点函数（重构：create_agent + 工具 + 结构化输出）
│   ├── graph.py         # 图构建（并行 fan-out、合并、fallback 边）
│   ├── config.py        # [新] LLM/重试/日志/Agent工具注册 配置
│   ├── tools.py         # [新] 知识检索 + 业务操作工具（初期 mock）
│   └── logger.py        # [新] 结构化日志
├── tests/
│   ├── test_nodes.py    # [新] 节点单元测试
│   └── test_graph.py    # [新] 图集成测试
├── test.py              # [保留] CLI 手动测试脚本
├── ultra.py             # [保留] 高级技巧参考
└── requirements.txt     # [新] 依赖声明
```

---

## 2. 状态定义（state.py）

### 结构化输出模型（Pydantic）

- `CategoryOutput`: `categories: list[Literal[...]]` + `confidence: float`
- `QualityOutput`: `score: int` (1-5) + `reason: str`
- `MergeOutput`: `final_response: str` + `sources: list[str]`

### AgentState

| 字段 | 类型 | 变化 |
|------|------|------|
| routing_targets | list[str] | 替代原 category: str，支持多指派 |
| assigned_agents | list[str] | [新] 实际激活的 Agent 列表 |
| agent_outputs | dict[str, str] | [新] {"technical": "回复", ...} |
| merged_response | str | [新] 合并后的最终回复 |
| quality_reason | str | [新] 评分理由 |

其余字段（conversation_id, messages, quality_score, requires_human_review, final_response, metadata）保持不变。

---

## 3. Agent 实现（node.py）

每个下游 Agent 使用 `langgraph.prebuilt.create_agent` 创建，各自挂载工具：

```
technical_agent → [search_product_docs, get_api_example]
sales_agent     → [query_pricing, check_promotion, search_product_docs]
support_agent   → [lookup_order, check_refund_status, search_product_docs]
feedback_agent  → [log_feedback_to_crm]
```

### 新增节点

| 节点 | 实现方式 | 职责 |
|------|---------|------|
| merge_agent | create_agent (无工具) | 合并多 Agent 输出，去重/统一语气 |
| fallback_reply | 纯函数 | 所有降级路径的终点，返回兜底回复 |

---

## 4. 图结构（graph.py）

```
categorize ──→ routing_targets 为空？──→ fallback_reply ──→ format_response

categorize ──→ routing_targets 有效 ──→ [条件边 + 并行 fan-out]
                    ├─ technical_agent ──┐
                    ├─ sales_agent     ──┤
                    ├─ support_agent   ──┤  (仅激活 routing_targets 中存在的 Agent)
                    └─ feedback_agent  ──┘
                                          ↓
                                    merge_agent
                                          ↓
                                    quality_check
                                       ╱    ╲
                              human_review  auto_approve
                                       ╲    ╱
                                    format_response
```

### 错误/降级边

| 触发条件 | 路由 |
|---------|------|
| routing_targets 为空 | categorize → fallback_reply → format_response |
| 所有 Agent 失败 | 汇聚 → fallback_reply → format_response |
| merge 失败 | merge → fallback_reply → format_response |

---

## 5. 配置（config.py）

纯数据，不包含逻辑：

- `LLM_CONFIG`: model, temperature
- `RETRY_POLICY`: max_attempts=3, initial_interval=1.0s, max_interval=30s
- `LOG_CONFIG`: level, format
- `AGENT_TOOLS`: 声明式工具注册表（名字到工具函数列表的映射）

---

## 6. 工具层（tools.py）

### 知识检索工具
- `search_product_docs(query)` — 搜索产品文档
- `get_api_example(endpoint)` — 获取 API 示例

### 业务操作工具
- `query_pricing(plan)` — 查询套餐价格
- `check_promotion(code)` — 校验优惠码
- `lookup_order(order_id)` — 查询订单状态
- `check_refund_status(refund_id)` — 查询退款进度
- `log_feedback_to_crm(content, contact)` — 记录反馈

初期用 mock 数据实现，接口签名稳定后替换为真实 API。

---

## 7. 提示词管理（prompts.py）

所有 prompt 模板集中为 `PROMPTS` 字典，按节点名索引。模板使用 `.format(**ctx)` 填入上下文。Agent 的 prompt 通过 `create_agent` 的 `system_prompt` 参数注入。

---

## 8. 合并策略

| 场景 | 策略 |
|------|------|
| 信息互补 | 按重要性排序，统一语气 |
| 信息重叠 | 去重，保留最详细版本 |
| 信息矛盾 | 提示差异，技术准确性优先 |
| 单 Agent | 直接采用，不画蛇添足 |

---

## 9. 日志

结构化日志，每条记录自动携带 `conversation_id` 和 `thread_id`。

埋点位置：各节点入口/出口、工具调用、路由决策、异常捕获。不记录含敏感信息的 prompt 原文。

---

## 10. 测试

pytest 自动化替代 print 手动验证：

- **单元测试**：categorize（单/多/空匹配）、merge（去重/冲突）、fallback
- **集成测试**：单 Agent 全链路、多 Agent 全链路、全失败降级、流式输出、多轮记忆

LLM 调用用 mock 避免依赖真实 API key。

---

## 11. 迁移路径

| 步骤 | 内容 |
|------|------|
| 1 | 新建 prompts.py，迁移现有 8 个 prompt |
| 2 | 新建 config.py，抽离 LLM/Retry/日志配置 |
| 3 | 新建 tools.py，mock 工具实现 |
| 4 | 改写 node.py：多分类 + create_agent + 工具 |
| 5 | 改写 graph.py：并行 fan-out + merge + fallback |
| 6 | 升级 state.py：新增路由字段 |
| 7 | 搭建 tests/ + pytest 自动化 |
