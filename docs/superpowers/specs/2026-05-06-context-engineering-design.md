# 上下文工程优化 — 设计文档

> **关联文档**: docs/上下文难题.md（问题1：任务越界幻觉；问题2：控制权转移错位）

**目标**: 在单流向多 Agent 架构下，解决 Supervisor 路由缺失上下文、Agent 间信息孤岛、以及架构冗余问题。

**架构**: 删除并行 fan-out + merge 节点，简化为单线流；新增 `conversation_log` 让 Supervisor 在路由时看到已结束轮次的对���摘要；每个业务 Agent 通过独立 MemorySaver 维护自身记忆。

**技术栈**: LangGraph 1.1.x, LangChain agents (create_agent), Pydantic, SQLAlchemy + SQLite

---

## 变更一：删除并行 fan-out + merge

### 问题

当前 `routing_targets: list[str]` 支持 Router 一次返回多个业务 intent，并行执行多个 Agent 后由 merge 合并。实际场景中几乎不会出现同一请求需要多个业务 Agent 并行处理的场景。并行 fan-out 和 merge 节点增加了 graph 复杂度但无实际收益。

### 方案

- `AgentState.routing_targets` 从 `list[str]` 改为 `routing_target: str`
- 删除 `merge_agent_outputs` 节点函数
- 删除 graph 中 `_BUSINESS_AGENTS` 的动态并行边、`route_after_agent` 条件边
- Router 只返回单个 intent
- 图结构变为：

```
supervisor_router → [conversation_agent | 单个业务Agent] → quality → format → END
                                   └─ 全部失败 → fallback → format → END
```

### 受影响的文件

- `core/state.py` — `routing_targets` 类型变更
- `core/node.py` — 删除 `merge_agent_outputs`，`supervisor_router` 返回变更
- `core/graph.py` — 简化图结构
- `core/prompts.py` — 删除 merge prompt

---

## 变更二：conversation_log 上下文传递

### 问题

`supervisor_router` 当前只看 `messages[-1]["content"]`（最后一条用户消息），完全不知道之前轮次发生了什么：

- Turn 1: "查订单" → support 处理完成
- Turn 2: "那退款吧" → Router 盲猜，不知道上一轮建立的是售后语境

### 方案

`AgentState` 新增 `conversation_log: list[dict]`，记录每轮已结束的对话摘要：

```python
# 字段结构
{"turn": 1, "agent": "support", "user": "查订单", "reply": "订单#12345已发货，预计2026-05-10到达"}
```

`supervisor_router` 输入从仅当前用户消息，扩展为：

```
当前用户消息: {messages[-1]["content"]}
历史轮次:
- [Turn 1] support: 用户问"查订单" → 回复"订单#12345已发货..."
- [Turn 2] sales: 用户问"有优惠吗" → 回复"目前没有活动..."
```

### 日志写入点

`format_response` 节点：每轮结束时，从 state 中提取 `messages[-1]["content"]`（本轮用户输入）、`final_response`（本轮 Agent 回复）、`intent`（路由目标），追加到 `conversation_log`。

### 不受污染

`conversation_log` 只存自然语言回复，不包含：
- 工具调用（tool call / tool result）
- Agent 内部提示词
- 中间推理过程

Supervisor 看到的是干净的对话摘要，不会产生文档描述的任务越界幻觉。

### 受影响的文件

- `core/state.py` — 新增 `conversation_log` 字段及类型
- `core/node.py` — `supervisor_router` 输入扩展，`format_response` 追加日志
- `core/prompts.py` — router prompt 增加 conversation_log 上下文段

---

## 变更三：下游 Agent 隔离记忆（已完成，仅记录）

### 方案

每个 Agent 在 `_build_agent()` 中绑定独立的 `MemorySaver()`，key 为 `{conv_id}_{agent_name}`。同一会话内同一 Agent 被 Router 多次命中时，能看到自己之前所有轮次的上下文。Agent 之间不共享记忆。

- 已实现于 `core/agent_factory.py`

---

## 不在此次范围

- **问题3（工具调用污染）** — 需要工具调用中间层，涉及工具架构重构，延后处理
- **回环模式** — 维持单向流，不引入 Agent → Portal 双向交接
- **跨 Agent 共享记忆** — 下游 Agent 不���要知道同一会话中其他 Agent 做了什么，维持隔离

---

## 变更后的完整图结构

```
supervisor_router ──┬── greeting/ambiguous → conversation_agent ──┐
                    ├── technical ──────────────────────────────┤
                    ├── sales ───────────────────────────────────┤
                    ├── support ─────────────────────────────────┤
                    └── feedback ────────────┬───────────────────┤
                                             └── fallback ───────┤
                                                                  ↓
                                                            quality_check
                                                            ↓           ↓
                                                      human_review  auto_approve
                                                            ↓           ↓
                                                      format_response ← END
```
