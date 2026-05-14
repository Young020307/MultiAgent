# 上下文工程优化 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除并行 fan-out + merge，新增 conversation_log 让 Supervisor 路由时看到对话历史摘要。

**Architecture:** 单向流：`supervisor_router → [conversation_agent | 单个业务Agent] → quality → format → END`。Router 从 `conversation_log` 获取前序轮次的干净摘要，每个 Agent 维护独立 MemorySaver 隔离记忆。

**Tech Stack:** LangGraph 1.1.x, LangChain agents (create_agent), Pydantic, SQLAlchemy + SQLite

---

## 文件结构

| 文件 | 职责 | 操作 |
|------|------|------|
| `core/state.py` | Pydantic 模型 + AgentState TypedDict | 修改 |
| `core/prompts.py` | 所有提示词模板 | 修改 |
| `core/node.py` | 图节点函数 | 修改 |
| `core/agent_factory.py` | Agent 工厂 + 节点包装 | 修改 |
| `core/graph.py` | 图构建 | 修改 |
| `chat.py` | 交互终端 | 修改 |
| `test.py` | CLI 批测脚本 | 修改 |

---

### Task 1: 更新 AgentState 类型定义

**Files:**
- Modify: `core/state.py`

- [ ] **Step 1: 改 routing_targets 为单数，删除并行相关字段，新增 conversation_log**

```python
# core/state.py — 完整替换 AgentState
class AgentState(TypedDict):
    conversation_id: str
    messages: Annotated[List[Message], operator.add]
    intent: str
    routing_target: str                              # ← list[str] → str
    merged_response: str
    quality_score: int
    quality_reason: str
    requires_human_review: bool
    final_response: str
    conversation_log: list[dict]                     # ← 新增
    metadata: dict
```

删除的字段：`routing_targets`、`assigned_agents`、`agent_outputs`。

- [ ] **Step 2: 验证语法**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "from core.state import AgentState; print('OK')"
```

---

### Task 2: 更新提示词

**Files:**
- Modify: `core/prompts.py`

- [ ] **Step 1: 删除 merge prompt，更新 router prompt**

```python
# core/prompts.py — 完整替换 PROMPTS
PROMPTS = {
    "router": """判断客户消息的意图类型。

{context}

当前客户消息：{user_message}

类型说明：
- greeting: 打招呼、闲聊、纯问候
- ambiguous: 有求助意图但具体需求模糊，需要澄清
- technical: 技术问题、产品使用、API 集成
- sales: 价格、购买、套餐、折扣
- support: 售后、退款、投诉、账户
- feedback: 建议、反馈、评价

严格返回 JSON（不要 markdown 包裹）：
{{"intent": "technical", "confidence": 0.9}}""",

    "conversation_agent": """你是智能客服主管，负责接待客户的问候、闲聊和模糊咨询。

你的职责：
- greeting（问候/闲聊）：友好回应，主动介绍你能提供的帮助（查订单、问优惠、技术问题等）
- ambiguous（模糊意图）：生成选择题帮客户快速消歧，例如 "请问您是需要 A.XXX B.XXX 还是 C.XXX？"
- 如果 clarify_count >= 2 仍无法明确意图：给出安全兜底话术，建议联系人工客服

客户消息中会附带 intent 和 clarify_count 信息，请据此做出恰当回应。""",

    "technical": """你是一位资深技术专家，专门解答产品使用和技术问题。
你的回答应该：
- 准确、详细、专业
- 提供具体的操作步骤
- 包含相关示例代码（如适用）
- 指出可能的注意事项和最佳实践

{context}

用户问题：{user_message}
请提供专业、详细的技术解答。""",

    "sales": """你是一位专业的销售顾问，负责解答价格、购买相关问题。
你的回答应该：
- 友好、热情、有说服力
- 清晰说明价格信息和优惠政策
- 主动推荐合适的产品套餐
- 提供购买链接或引导

{context}

用户咨询：{user_message}
请提供有帮助的销售建议。""",

    "support": """你是一位耐心的售后支持专家，处理用户投诉、退款、账户问题。
你的回答应该：
- 表达理解和同理心
- 提供清晰的解决方案
- 说明处理流程和预计时间
- 必要时提供升级渠道

{context}

用户问题：{user_message}
请提供周到、专业的支持回复。""",

    "feedback": """你负责处理用户反馈和建议。
你的回答应该：
- 感谢用户的反馈
- 认真对待每一条建议
- 说明反馈的处理流程
- 邀请用户继续参与产品改进

用户反馈：{user_message}
请提供真诚、专业的回复。""",

    "quality": """评估以下客服回复的质量。

咨询类别：{categories}
客服回复：{response}

评分标准（1-5）：
5-完美 4-良好 3-一般 2-较差 1-不合格

严格按以下 JSON 格式返回（不要用 markdown 代码块包裹）：
{{"score": 4, "reason": "回复准确但可以更详细"}}""",
}
```

变化：
- `router` 增加 `{context}` 占位符，插入 conversation_log 历史摘要
- 删除 `merge` prompt
- 删除业务 Agent prompt 中残留的 `{context}` 占位符（当前未使用）

- [ ] **Step 2: 验证语法**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "from core.prompts import PROMPTS; print(len(PROMPTS)); assert 'merge' not in PROMPTS; print('OK')"
```

---

### Task 3: 更新 supervisor_router 节点

**Files:**
- Modify: `core/node.py`（supervisor_router 函数）

- [ ] **Step 1: 改 routing_targets → routing_target，构建 conversation_log 上下文**

```python
# core/node.py — 完整替换 supervisor_router
def supervisor_router(state: AgentState) -> dict:
    """纯路由决策 — 只输出 intent，由条件边决定下一步。"""
    messages = state["messages"]
    user_message = messages[-1]["content"]
    meta = state.get("metadata", {})

    logger.info("[router] 意图判断 conversation=%s len=%d",
                state.get("conversation_id", "?"), len(user_message))

    # 构建 conversation_log 上下文
    log = state.get("conversation_log", [])
    if log:
        lines = []
        for entry in log:
            lines.append(
                f"- [Turn {entry['turn']}] {entry['agent']}: "
                f"用户问"{entry['user']}" → 回复"{entry['reply'][:80]}...""
            )
        context = "历史轮次:\n" + "\n".join(lines)
    else:
        context = "历史轮次: （首轮对话）"

    prompt = PROMPTS["router"].format(user_message=user_message, context=context)
    raw = get_llm_for("supervisor_router").invoke(prompt).content
    decision = parse_json(raw, RouterDecision)

    intent = decision.intent
    business = ["technical", "sales", "support", "feedback"]

    if intent in business:
        logger.info("[router] 业务路由 intent=%s confidence=%.2f", intent, decision.confidence)
        return {
            "intent": intent,
            "routing_target": intent,
            "metadata": {**meta, "router_intent": intent, "clarify_count": 0},
        }

    logger.info("[router] 对话路由 intent=%s", intent)
    return {
        "intent": intent,
        "routing_target": "conversation_agent",
        "metadata": {**meta, "router_intent": intent},
    }
```

变化：
- `routing_targets: [intent]` → `routing_target: intent`
- `routing_targets: []` → `routing_target: "conversation_agent"`
- 新增 conversation_log 格式化后填入 prompt 的 `{context}`

- [ ] **Step 2: 验证语法**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "from core.node import supervisor_router; print('OK')"
```

---

### Task 4: 更新 format_response + 删除 merge_agent_outputs

**Files:**
- Modify: `core/node.py`（format_response 函数，删除 merge_agent_outputs 函数）

- [ ] **Step 1: 删除 merge_agent_outputs，更新 format_response**

删除整个 `merge_agent_outputs` 函数（当前约 20 行）。替换 `format_response`：

```python
# core/node.py — 完整替换 format_response
def format_response(state: AgentState) -> dict:
    final = state.get("final_response", "")
    formatted = (
        f"{final}\n"
        f"---\n"
        f"**客服团队** | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"工单编号：{state.get('conversation_id', 'N/A')}"
    )

    # 追加 conversation_log
    messages = state.get("messages", [])
    user_msg = messages[-1]["content"] if messages else ""
    log = state.get("conversation_log", [])
    turn = len(log) + 1
    new_entry = {
        "turn": turn,
        "agent": state.get("routing_target", state.get("intent", "unknown")),
        "user": user_msg,
        "reply": state.get("final_response", ""),
    }

    return {
        "final_response": formatted,
        "conversation_log": [*log, new_entry],
    }
```

- [ ] **Step 2: 删除 quality_check / human_review / auto_approve 中对已删除 state 字段的引用**

`quality_check`、`human_review`、`auto_approve`、`fallback_reply` 不需要修改（它们不引用 `routing_targets`/`agent_outputs`/`assigned_agents`）。确认 `quality_check` 读的是 `merged_response`（保留）。

- [ ] **Step 3: 验证语法**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "from core.node import format_response, supervisor_router; print('OK')"
```

---

### Task 5: 更新 wrap_agent_node（业务 Agent 直接写 merged_response）

**Files:**
- Modify: `core/agent_factory.py`（wrap_agent_node 函数中业务 Agent 的返回）

- [ ] **Step 1: 业务 Agent 返回 merged_response 替代 agent_outputs**

```python
# core/agent_factory.py — wrap_agent_node 中业务 Agent 的 return 部分替换为：
        return {"merged_response": response_text}
```

（conversation_agent 分支不变，仍返回 `final_response`。）

完整变更位置：`agent_factory.py:169-172`，将原来的：

```python
        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: response_text},
            "assigned_agents": [*state.get("assigned_agents", []), agent_name],
        }
```

替换为：

```python
        return {"merged_response": response_text}
```

- [ ] **Step 2: 验证语法**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "from core.agent_factory import wrap_agent_node; print('OK')"
```

---

### Task 6: 简化 graph.py 图结构

**Files:**
- Modify: `core/graph.py`

- [ ] **Step 1: 重写 graph.py**

```python
# core/graph.py
"""图构建：

supervisor_router（纯路由决策）
  ├─ intent=greeting/ambiguous → conversation_agent → format_response → END
  └─ intent=business → 业务Agent → quality → human/auto → format → END
       └─ 响应为空 → fallback → format → END
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from core.node import (
    supervisor_router,
    fallback_reply,
    quality_check,
    human_review,
    auto_approve,
    format_response,
)
from core.agent_factory import wrap_agent_node
from core.logger import setup_logger

logger = setup_logger("graph")
memory = MemorySaver()

_BUSINESS_AGENTS = ("technical", "sales", "support", "feedback")


# ============ 路由函数 ============

def route_after_router(state: AgentState) -> str:
    """supervisor_router 后的条件路由。"""
    target = state.get("routing_target", "conversation_agent")

    if target in _BUSINESS_AGENTS:
        logger.info("[route] intent=%s → %s", state.get("intent"), target)
        return target

    logger.info("[route] intent=%s → conversation_agent", state.get("intent"))
    return "conversation_agent"


def route_after_agent(state: AgentState) -> str:
    """业务 Agent 后：有回复走质检，无回复走兜底。"""
    return "quality_check" if state.get("merged_response") else "fallback"


def route_after_quality(state: AgentState) -> str:
    if state.get("requires_human_review", False):
        return "human_review"
    return "auto_approve"


# ============ 构建图 ============

customer_service_graph = StateGraph(AgentState)

# 节点注册
customer_service_graph.add_node("supervisor_router", supervisor_router)
customer_service_graph.add_node("conversation_agent", wrap_agent_node("conversation_agent"))
for name in _BUSINESS_AGENTS:
    customer_service_graph.add_node(name, wrap_agent_node(name))
customer_service_graph.add_node("fallback", fallback_reply)
customer_service_graph.add_node("quality_check", quality_check)
customer_service_graph.add_node("human_review", human_review)
customer_service_graph.add_node("auto_approve", auto_approve)
customer_service_graph.add_node("format_response", format_response)

customer_service_graph.set_entry_point("supervisor_router")

# supervisor_router → conversation_agent 或 单个业务 Agent
customer_service_graph.add_conditional_edges(
    "supervisor_router",
    route_after_router,
    {
        "conversation_agent": "conversation_agent",
        "technical": "technical",
        "sales": "sales",
        "support": "support",
        "feedback": "feedback",
    },
)

# conversation_agent 直接回复 → format_response（跳过 merge/quality）
customer_service_graph.add_edge("conversation_agent", "format_response")

# 业务 Agent → quality 或 fallback
for name in _BUSINESS_AGENTS:
    customer_service_graph.add_conditional_edges(
        name,
        route_after_agent,
        {"quality_check": "quality_check", "fallback": "fallback"},
    )

# quality → human/auto → format
customer_service_graph.add_conditional_edges(
    "quality_check",
    route_after_quality,
    {"human_review": "human_review", "auto_approve": "auto_approve"},
)
customer_service_graph.add_edge("human_review", "format_response")
customer_service_graph.add_edge("auto_approve", "format_response")

# fallback → format（跳过审核）
customer_service_graph.add_edge("fallback", "format_response")

customer_service_graph.add_edge("format_response", END)

customer_service_app = customer_service_graph.compile(checkpointer=memory)
```

变化：
- 删除 `merge_agent_outputs` import
- 删除 merge 节点注册
- `route_after_router` 返回 `str` 而非 `list[str]`
- 条件边 map 删除 `"fallback": "fallback"`（无效路由不再可能，因为 `routing_target` 总是有效值）
- 删除 `merge → quality` 边，改为各业务 Agent → quality/fallback

- [ ] **Step 2: 验证语法和编译**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "from core.graph import customer_service_app; print('graph compiled OK')"
```

---

### Task 7: 更新 chat.py 显示逻辑

**Files:**
- Modify: `chat.py`

- [ ] **Step 1: 更新 build_initial_state 和显示逻辑**

```python
# chat.py — 完整替换
#!/usr/bin/env python
"""交互式终端对话 — 与客服 Agent 系统实时交流。"""

import uuid
from datetime import datetime
from core.graph import customer_service_app


def main():
    conversation_id = str(uuid.uuid4())[:8]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 55)
    print("  客服 Agent 系统 — 交互模式")
    print("  输入消息开始对话，quit/exit/q 退出，clear 清除记忆")
    print("=" * 55)

    def build_state(message: str) -> dict:
        return {
            "conversation_id": conversation_id,
            "messages": [{"role": "user", "content": message, "timestamp": datetime.now().isoformat()}],
            "intent": "",
            "routing_target": "",
            "merged_response": "",
            "quality_score": 0,
            "quality_reason": "",
            "requires_human_review": False,
            "final_response": "",
            "conversation_log": [],
            "metadata": {},
        }

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见。")
            break
        if user_input.lower() == "clear":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print("[记忆已清除]")
            continue

        print()
        try:
            for event in customer_service_app.stream(
                build_state(user_input), config=config, stream_mode="updates"
            ):
                for node_name, output in event.items():
                    if node_name == "supervisor_router":
                        target = output.get("routing_target", "")
                        intent = output.get("intent", "")
                        print(f"[路由] intent={intent} → {target}")

                    elif node_name == "conversation_agent":
                        reply = output.get("final_response", "")
                        print(f"[对话Agent] 回复 ({len(reply)} 字符)")

                    elif node_name in ("technical", "sales", "support", "feedback"):
                        resp = output.get("merged_response", "")
                        print(f"  [{node_name}] 答复完成 ({len(resp)} 字符)")

                    elif node_name == "quality_check":
                        score = output["quality_score"]
                        review = "人工审核" if output.get("requires_human_review") else "通过"
                        print(f"  [质检] {score}/5 ({review})")

                    elif node_name == "format_response":
                        print()
                        print("-" * 40)
                        print(output.get("final_response", ""))
                        print("-" * 40)

        except Exception as e:
            print(f"[错误] {e}")


if __name__ == "__main__":
    main()
```

变化：
- `build_initial_state`：`routing_targets: []` → `routing_target: ""`，新增 `conversation_log: []`，删除 `assigned_agents: []`、`agent_outputs: {}`
- `supervisor_router` 显示：`routing_targets` → `routing_target`
- 业务 Agent 显示：从 `output["agent_outputs"]`/`assigned_agents` 改为 `output["merged_response"]`
- 删除 merge 节点显示分支

- [ ] **Step 2: 验证语法**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "from chat import main; print('OK')"
```

---

### Task 8: 更新 test.py

**Files:**
- Modify: `test.py`

- [ ] **Step 1: 更新 build_initial_state 和 stream_chat**

```python
# test.py — 完整替换
"""CLI 手动测试脚本 — 演示所有场景。"""

from datetime import datetime
import uuid
from langgraph.errors import GraphRecursionError
from core.graph import customer_service_app


def build_initial_state(message: str) -> dict:
    return {
        "conversation_id": str(uuid.uuid4())[:8],
        "messages": [{
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }],
        "routing_target": "",
        "merged_response": "",
        "quality_score": 0,
        "quality_reason": "",
        "requires_human_review": False,
        "final_response": "",
        "conversation_log": [],
        "metadata": {},
    }


def chat_with_customer(message: str, thread_id: str = None) -> str:
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = customer_service_app.invoke(
            build_initial_state(message),
            config=config,
        )
        return result["final_response"]
    except GraphRecursionError:
        return "[系统提示] 处理步骤过多，请简化问题后重试。"
    except Exception:
        return "[系统提示] 服务暂时不可用，已转接人工客服。"


def stream_chat(message: str, thread_id: str = None):
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f">> 开始处理：{message[:30]}...")
    print("-" * 40)
    try:
        for event in customer_service_app.stream(
            build_initial_state(message),
            config=config,
            stream_mode="updates",
        ):
            for node_name, output in event.items():
                if "merged_response" in output:
                    resp = output["merged_response"]
                    print(f"[{node_name}] 答复完成 ({len(resp)} 字符)")
                elif "quality_score" in output:
                    review = "人工审核" if output.get("requires_human_review") else "通过"
                    print(f"[{node_name}] 评分: {output['quality_score']}/5 ({review})")
                elif "final_response" in output:
                    print(f"[{node_name}] 最终回复已格式化")
        print("-" * 40)
    except Exception as e:
        print(f"[错误] {e}")


if __name__ == "__main__":
    # 场景 1-4：基础单轮
    tests = [
        ("技术问题", "如何配置 API 密钥？我的请求一直返回 401 错误。"),
        ("销售咨询", "企业版套餐有什么优惠？我们公司有 50 人需要购买。"),
        ("售后支持", "我昨天提交的退款申请什么时候能处理？"),
        ("反馈建议", "你们的APP很好用，建议增加深色模式。"),
    ]
    for i, (label, msg) in enumerate(tests, 1):
        print("=" * 50)
        print(f"场景 {i}：{label}")
        print("=" * 50)
        print(chat_with_customer(msg))
        print()

    # 场景 5：流式输出
    print("=" * 50)
    print("场景 5：流式输出")
    print("=" * 50)
    stream_chat("我的账号被锁定了，如何解锁？")
    print()

    # 场景 6：多轮对话
    print("=" * 50)
    print("场景 6：多轮对话（验证 conversation_log + 记忆）")
    print("=" * 50)
    thread = str(uuid.uuid4())
    for i, msg in enumerate([
        "你好，我想了解一下你们的API产品",
        "它的定价是多少？",
        "帮我对比一下基础版和企业版",
    ], 1):
        print(f"[第{i}轮]")
        r = chat_with_customer(msg, thread_id=thread)
        print(r[:200] + "...\n")
```

变化：
- `build_initial_state` 与 chat.py 同样的字段更新
- 删除 `stream_chat` 中的 `assigned_agents` 检查
- 删除 `merge` 相关的显示逻辑
- 场景 5 消息简化（不再需要"同时"多意图）

- [ ] **Step 2: 验证语法**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "from test import build_initial_state, chat_with_customer; print('OK')"
```

---

### Task 9: 集成测试

- [ ] **Step 1: 运行完整语法检查**

```bash
/home/neousys/anaconda3/envs/agent/bin/python -c "
from core.state import AgentState, RouterDecision, QualityOutput
from core.prompts import PROMPTS
from core.node import supervisor_router, fallback_reply, quality_check, human_review, auto_approve, format_response
from core.agent_factory import get_llm_for, get_agent, wrap_agent_node
from core.graph import customer_service_app
print('All imports OK')
print('Graph nodes:', list(customer_service_app.get_graph().nodes.keys()))
"
```

Expected: `All imports OK` + 节点列表不包含 `merge`

- [ ] **Step 2: 运行批测（需要 Ollama + MiniMax API 可用）**

```bash
cd /home/neousys/桌面/MultiAgent && /home/neousys/anaconda3/envs/agent/bin/python test.py
```

Expected: 6 个场景全部完成，无异常。场景 6 多轮对话中，第二轮 Router 能看到第一轮的 conversation_log。

- [ ] **Step 3: 验证 conversation_log 生效**

在 `test.py` 运行后，检查日志中 Router 的输出来确认 conversation_log 被正确传递。预期 `[router]` 日志行包含历史轮次信息。

或者修改 test.py 场景 6，在每轮后打印 `conversation_log` 长度：

```python
# 在 chat_with_customer 返回后加一行调试
result = customer_service_app.invoke(...)
print(f"[conversation_log 已累积 {len(result.get('conversation_log', []))} 条]")
```
