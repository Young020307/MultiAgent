# 人工介入（Human Handoff）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现三大人工触发场景：用户要求转人工、系统故障兜底、负面情感升级，通过 human_handoff 节点打包上下文并终止自动化流程。

**Architecture:** supervisor_router 新增 escalate 意图识别 + 关键词情感检测 → route_after_router 检查三大触发条件 → human_handoff 打包 handoff_context → END。anomaly_counters 在 format_response 中更新，成功重置/失败累加。

**Tech Stack:** 纯 Python，无新增依赖。情感检测用关键词匹配（零延迟）。

---

### Task 1: Update `core/state.py` — new fields and escalate intent

**Files:**
- Modify: `core/state.py`

- [ ] **Step 1: Add `escalate` to IntentType and new fields to AgentState**

Change `IntentType`:
```python
IntentType = Literal["greeting", "ambiguous", "technical", "sales", "support", "feedback", "escalate"]
```

In `AgentState` TypedDict, add these new fields after `conversation_log`:
```python
    escalated: bool
    escalate_reason: str
    handoff_context: dict
    anomaly_counters: dict
    sentiment: str
```

Full updated `AgentState`:
```python
class AgentState(TypedDict):
    conversation_id: str
    messages: Annotated[List[Message], operator.add]
    intent: IntentType
    routing_target: str
    merged_response: str
    quality_score: int
    quality_reason: str
    requires_human_review: bool
    final_response: str
    conversation_log: list[LogEntry]
    escalated: bool
    escalate_reason: str
    handoff_context: dict
    anomaly_counters: dict
    sentiment: str
    metadata: dict
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.state import AgentState, IntentType; print('OK')"`
Expected: `OK`



---

### Task 2: Add sentiment patterns and thresholds to `core/config.py`

**Files:**
- Modify: `core/config.py`

- [ ] **Step 1: Append constants**

Append to the end of `core/config.py`:

```python
# ================================================================
# 人工介入配置
# ================================================================

CRITICAL_PATTERNS = [
    "投诉", "举报", "忍无可忍", "我要告", "再也不用了",
    "骗子", "垃圾公司", "太失望了", "糟透了",
]
NEGATIVE_PATTERNS = [
    "不满意", "太差了", "很生气", "火大", "什么玩意",
    "糊弄", "忽悠", "扯淡", "坑人",
]

ANOMALY_THRESHOLDS = {
    "low_conf_streak": 2,        # 连续 low_confidence 达到此值触发
    "quality_fail_streak": 2,     # 连续 quality_score < 3 达到此值触发
    "tool_error_streak": 3,       # 连续工具错误达到此值触发
}
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.config import CRITICAL_PATTERNS, ANOMALY_THRESHOLDS; print(len(CRITICAL_PATTERNS), ANOMALY_THRESHOLDS)"`
Expected: `9 {'low_conf_streak': 2, 'quality_fail_streak': 2, 'tool_error_streak': 3}`



---

### Task 3: Update `core/prompts.py` — add escalate to router prompt

**Files:**
- Modify: `core/prompts.py`

- [ ] **Step 1: Add escalate type to router prompt**

In PROMPTS["router"], add `- escalate: 用户明确要求转人工，如"转人工"、"我要跟真人说话"、"有没有人工客服"` after the `- feedback` line.

The edit — change the router prompt. Replace:
```python
- feedback: 建议、反馈、评价
```
with:
```python
- feedback: 建议、反馈、评价
- escalate: 用户明确要求转人工，如"转人工"、"我要跟真人说话"、"有没有人工客服"
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.prompts import PROMPTS; assert 'escalate' in PROMPTS['router']; print('OK')"`
Expected: `OK`



---

### Task 4: Update `core/node.py` — sentiment detection + human_handoff node + counter management

**Files:**
- Modify: `core/node.py`

- [ ] **Step 1: Add sentiment detection and escalate/anomaly logic to supervisor_router**

Add imports at top:
```python
from core.config import CRITICAL_PATTERNS, NEGATIVE_PATTERNS, ANOMALY_THRESHOLDS
```

Add sentiment detection function (in the file, before `supervisor_router`):
```python
def _detect_sentiment(text: str) -> str:
    """关键词匹配情感检测。返回 "critical" | "negative" | "neutral"。"""
    text_lower = text.lower()
    for pattern in CRITICAL_PATTERNS:
        if pattern in text_lower:
            logger.info("[router] 检测到关键负面情感: %s", pattern)
            return "critical"
    for pattern in NEGATIVE_PATTERNS:
        if pattern in text_lower:
            return "negative"
    return "neutral"
```

Add anomaly check function:
```python
def _check_anomaly_thresholds(counters: dict) -> str | None:
    """检查异常计数器是否超阈值。返回触发原因或 None。"""
    thresholds = ANOMALY_THRESHOLDS
    for key, threshold in thresholds.items():
        if counters.get(key, 0) >= threshold:
            return f"system_failure:{key}"
    return None
```

In `supervisor_router`, after getting `decision`, add sentiment detection and anomaly check BEFORE returning business routing:

After `intent = decision.intent`:
```python
    # 情感检测
    sentiment = _detect_sentiment(user_message)

    # 异常计数器检查（上一轮遗留）
    anomaly_counters = state.get("anomaly_counters", {})
    anomaly_trigger = _check_anomaly_thresholds(anomaly_counters)

    # 场景 1：用户要求转人工
    if intent == "escalate":
        logger.info("[router] 转人工 - 用户主动请求")
        return {
            "intent": intent,
            "routing_target": "human_handoff",
            "escalated": True,
            "escalate_reason": "user_requested",
            "sentiment": sentiment,
            "metadata": {**meta, "router_intent": intent},
        }

    # 场景 5：负面情感升级
    if sentiment == "critical":
        logger.info("[router] 转人工 - 负面情感升级")
        return {
            "intent": intent,
            "routing_target": "human_handoff",
            "escalated": True,
            "escalate_reason": "negative_sentiment",
            "sentiment": sentiment,
            "metadata": {**meta, "router_intent": intent},
        }

    # 场景 4：系统故障兜底
    if anomaly_trigger:
        logger.info("[router] 转人工 - 系统异常: %s", anomaly_trigger)
        return {
            "intent": intent,
            "routing_target": "human_handoff",
            "escalated": True,
            "escalate_reason": "system_failure",
            "anomaly_counters": anomaly_counters,
            "sentiment": sentiment,
            "metadata": {**meta, "router_intent": intent},
        }
```

And add `sentiment` to the normal business routing return dicts:
```python
        "sentiment": sentiment,
```

- [ ] **Step 2: Add human_handoff node function**

```python
def human_handoff(state: AgentState) -> dict:
    """打包人工交接上下文，终止自动化流程。"""
    from datetime import datetime

    reason = state.get("escalate_reason", "unknown")
    messages = state.get("messages", [])

    handoff = {
        "reason": reason,
        "last_messages": messages[-5:],
        "conversation_log": state.get("conversation_log", [])[-3:],
        "intent": state.get("intent"),
        "sentiment": state.get("sentiment", "neutral"),
        "anomaly_counters": state.get("anomaly_counters", {}),
        "timestamp": datetime.now().isoformat(),
    }

    user_notice = {
        "user_requested": "已为您转接人工客服，请稍候...",
        "system_failure": "系统暂时无法处理您的请求，正在为您转接人工客服...",
        "negative_sentiment": "我们非常重视您的体验，已为您转接高级人工客服，请稍候...",
    }.get(reason, "正在为您转接人工客服，请稍候...")

    logger.info("[handoff] 人工交接 reason=%s", reason)
    return {
        "final_response": user_notice,
        "escalated": True,
        "escalate_reason": reason,
        "handoff_context": handoff,
        "metadata": {**state.get("metadata", {}), "human_handoff": True},
    }
```

- [ ] **Step 3: Update format_response to manage anomaly_counters**

In `format_response`, add counter management logic before the return. After computing `new_entry`, add:

```python
    # 异常计数器管理：成功回复则重置，失败则已在 router 前累积
    quality_score = state.get("quality_score", 0)
    requires_human = state.get("requires_human_review", False)

    # 从上一轮的计数器基础上更新
    counters = {**state.get("anomaly_counters", {})}

    # 成功回复 → 重置所有计数器
    if not requires_human:
        counters = {"low_conf_streak": 0, "quality_fail_streak": 0, "tool_error_streak": 0}
    else:
        # 质检失败 → 累加
        counters["quality_fail_streak"] = counters.get("quality_fail_streak", 0) + 1

    # low_conf 累加由 tools.py 的 search_knowledge_base 负责（通过 _consecutive_low_conf）
    # 但这里也需要同步到 state 中供 router 检查
```

实际上需要从 `core.tools` 中读取低置信度状态。更好的做法是在 `format_response` 末尾检查上一轮的 metadata 或 merged_response 中是否包含低置信度标记。

Simpler approach: 在 `format_response` 结尾统一读取上一轮计数器，然后：
```python
    # 从上一轮继承计数器并管理
    old_counters = state.get("anomaly_counters", {})
    counters = {**old_counters}

    # quality 检测
    if state.get("quality_score", 5) < 3:
        counters["quality_fail_streak"] = counters.get("quality_fail_streak", 0) + 1
    else:
        counters["quality_fail_streak"] = 0

    # 成功生成回复且无异常 → 重置
    merged = state.get("merged_response", "")
    if merged and not state.get("requires_human_review", False):
        counters["quality_fail_streak"] = 0
```

Combine with tools.py's low_conf streak: we need a way to pass it to state. Simplest: store `_consecutive_low_conf` value in metadata during `search_knowledge_base`, then `format_response` reads it.

Actually, let me simplify: `format_response` adds the counter values to the returned `anomaly_counters`. The key low_conf_streak comes from the tools module via metadata. Let me just define the update logic clearly.

Final approach for `format_response` counter update:
```python
    counters = {**state.get("anomaly_counters", {})}
    
    # 从 metadata 读取本轮的低置信度状态（由 search_knowledge_base 写入）
    was_low_conf = state.get("metadata", {}).get("rag_low_confidence", False)
    if was_low_conf:
        counters["low_conf_streak"] = counters.get("low_conf_streak", 0) + 1
    else:
        counters["low_conf_streak"] = 0
    
    # 质检评级
    if state.get("quality_score", 5) < 3:
        counters["quality_fail_streak"] = counters.get("quality_fail_streak", 0) + 1
    else:
        counters["quality_fail_streak"] = 0
    
    # 正常完成 → 重置（只有生成了有效回复才算）
    if state.get("final_response") or state.get("merged_response"):
        pass  # 计数器已经被上面更新了，保留
```

And in `search_knowledge_base` (tools.py), we add a line to set metadata flag.

Wait, this is getting complex within the plan. Let me simplify:

In `core/tools.py`, in `search_knowledge_base`, after calling `_rag_client.search()` and tracking counts, also add a flag to the result JSON or set a module-level variable that `format_response` can read.

Actually the simplest approach: `format_response` already has access to `state["metadata"]`. In `search_knowledge_base` we can't write to state directly (it's a tool), but we can expose the low_conf state via a module-level getter. Then `format_response` calls it.

Actually the SIMPLEST: just have `format_response` update counters based only on `quality_score` and `requires_human_review` from state, which it already has. The `low_conf_streak` counter... hmm, we need it from the search tool.

OK let me just define a `get_low_conf_streak(conversation_id)` function in tools.py that `format_response` can call. That's clean.

- [ ] **Step 4: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.node import supervisor_router, human_handoff, format_response; print('OK')"`
Expected: `OK`



---

### Task 5: Update `core/graph.py` — register human_handoff and routing

**Files:**
- Modify: `core/graph.py`

- [ ] **Step 1: Update imports and add human_handoff node**

In the import from node, add `human_handoff`:
```python
from core.node import (
    supervisor_router,
    fallback_reply,
    quality_check,
    human_review,
    auto_approve,
    format_response,
    human_handoff,
)
```

Add the node:
```python
customer_service_graph.add_node("human_handoff", human_handoff)
```

- [ ] **Step 2: Update route_after_router**

```python
def route_after_router(state: AgentState) -> str:
    """supervisor_router 后的条件路由。"""
    # 人工交接直接路由
    if state.get("routing_target") == "human_handoff":
        logger.info("[route] → human_handoff reason=%s", state.get("escalate_reason"))
        return "human_handoff"

    target = state.get("routing_target", "conversation_agent")
    if target in _BUSINESS_AGENTS:
        logger.info("[route] intent=%s → %s", state.get("intent"), target)
        return target

    logger.info("[route] intent=%s → conversation_agent", state.get("intent"))
    return "conversation_agent"
```

- [ ] **Step 3: Update conditional edges mapping and add human_handoff → END**

In `add_conditional_edges("supervisor_router", ...)`, add `"human_handoff": "human_handoff"` to the mapping dict:
```python
customer_service_graph.add_conditional_edges(
    "supervisor_router",
    route_after_router,
    {
        "conversation_agent": "conversation_agent",
        "technical": "technical",
        "sales": "sales",
        "support": "support",
        "feedback": "feedback",
        "human_handoff": "human_handoff",
    },
)
```

Add:
```python
customer_service_graph.add_edge("human_handoff", END)
```

- [ ] **Step 4: Add interrupt_before to graph compilation**

Change the `customer_service_app = customer_service_graph.compile(checkpointer=memory)` line to:
```python
customer_service_app = customer_service_graph.compile(
    checkpointer=memory,
    interrupt_before=["human_handoff"],
)
```

- [ ] **Step 5: Verify graph compiles with interrupt**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.graph import customer_service_app; print('Graph OK')"`
Expected: `Graph OK`



---

### Task 6: Update `core/tools.py` — expose low_conf streak to state

**Files:**
- Modify: `core/tools.py`

- [ ] **Step 1: Add functions to read/clear low_conf streak from format_response**

Add at the end of the RAG section:
```python
def get_low_conf_streak(conversation_id: str) -> int:
    """供 format_response 读取当前低置信度连续次数。"""
    return _consecutive_low_conf.get(conversation_id, 0)


def reset_low_conf_streak(conversation_id: str):
    """成功回复后重置低置信度计数。"""
    if conversation_id in _consecutive_low_conf:
        _consecutive_low_conf[conversation_id] = 0
```

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.tools import get_low_conf_streak, reset_low_conf_streak; print('OK')"`
Expected: `OK`



---

### Task 7: Update `core/node.py:format_response` — wire counter management

**Files:**
- Modify: `core/node.py` (format_response function only)

- [ ] **Step 1: Update format_response to manage anomaly_counters**

Add import at top of node.py:
```python
from core.tools import get_low_conf_streak, reset_low_conf_streak
```

In `format_response`, after building `new_entry` but before the return, add counter management:

```python
    conv_id = state.get("conversation_id", "")
    counters = {**state.get("anomaly_counters", {})}

    # low_conf 连续次数（从 tools.py 读取）
    low_streak = get_low_conf_streak(conv_id) if conv_id else 0
    counters["low_conf_streak"] = low_streak

    # 质检失败计数
    if state.get("quality_score", 5) < 3:
        counters["quality_fail_streak"] = counters.get("quality_fail_streak", 0) + 1
    else:
        counters["quality_fail_streak"] = 0

    # 生成了有效回复 → 重置 low_conf 计数器
    if state.get("merged_response") and not state.get("requires_human_review", False):
        if conv_id:
            reset_low_conf_streak(conv_id)
        counters["quality_fail_streak"] = 0
```

And add `"anomaly_counters": counters` to the return dict.

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "from core.node import format_response; print('OK')"`
Expected: `OK`



---

### Task 8: Integration test — verify interrupt routing + no regression

**Files:**
- No files created (verification only)

- [ ] **Step 1: Test Scenario 1 — escalate via user request (interrupts before human_handoff)**

Run:
```bash
PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "
from datetime import datetime
from core.graph import customer_service_app

state = {
    'conversation_id': 'test-escalate',
    'messages': [{'role': 'user', 'content': '我要转人工', 'timestamp': datetime.now().isoformat()}],
    'routing_target': '', 'merged_response': '', 'quality_score': 0, 'quality_reason': '',
    'requires_human_review': False, 'final_response': '', 'conversation_log': [],
    'escalated': False, 'escalate_reason': '', 'handoff_context': {},
    'anomaly_counters': {}, 'sentiment': 'neutral', 'metadata': {},
}
cfg = {'configurable': {'thread_id': 'test-escalate'}}
# interrupt_before 会在 human_handoff 前挂起，invoke 返回中断前的状态
result = customer_service_app.invoke(state, config=cfg)
assert result['routing_target'] == 'human_handoff', f'Expected human_handoff, got {result[\"routing_target\"]}'
assert result['escalated'] == True
assert result['escalate_reason'] == 'user_requested'
print('Scenario 1 PASS: router sets escalate correctly, graph interrupted before human_handoff')
"
```
Expected: `Scenario 1 PASS`

- [ ] **Step 2: Test Scenario 5 — negative sentiment triggers handoff**

Run:
```bash
PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "
from datetime import datetime
from core.graph import customer_service_app

state = {
    'conversation_id': 'test-sentiment',
    'messages': [{'role': 'user', 'content': '你们这个服务太差了，我要投诉！骗子！', 'timestamp': datetime.now().isoformat()}],
    'routing_target': '', 'merged_response': '', 'quality_score': 0, 'quality_reason': '',
    'requires_human_review': False, 'final_response': '', 'conversation_log': [],
    'escalated': False, 'escalate_reason': '', 'handoff_context': {},
    'anomaly_counters': {}, 'sentiment': 'neutral', 'metadata': {},
}
cfg = {'configurable': {'thread_id': 'test-sentiment'}}
result = customer_service_app.invoke(state, config=cfg)
assert result.get('escalated') == True
assert result.get('escalate_reason') == 'negative_sentiment'
print('Scenario 5 PASS: sentiment triggers handoff, graph interrupted')
"
```
Expected: `Scenario 5 PASS`

- [ ] **Step 3: Test Scenario 4 — anomaly counters trigger handoff**

Run:
```bash
PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "
from datetime import datetime
from core.graph import customer_service_app

state = {
    'conversation_id': 'test-anomaly',
    'messages': [{'role': 'user', 'content': '退款', 'timestamp': datetime.now().isoformat()}],
    'routing_target': '', 'merged_response': '', 'quality_score': 0, 'quality_reason': '',
    'requires_human_review': False, 'final_response': '', 'conversation_log': [],
    'escalated': False, 'escalate_reason': '', 'handoff_context': {},
    'anomaly_counters': {'low_conf_streak': 2}, 'sentiment': 'neutral', 'metadata': {},
}
cfg = {'configurable': {'thread_id': 'test-anomaly'}}
result = customer_service_app.invoke(state, config=cfg)
assert result.get('escalated') == True
assert result.get('escalate_reason') == 'system_failure'
print('Scenario 4 PASS: anomaly counters trigger handoff, graph interrupted')
"
```
Expected: `Scenario 4 PASS`

- [ ] **Step 4: Test interrupt + resume + context injection**

Run:
```bash
PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "
from datetime import datetime
from core.graph import customer_service_app

# 1. 触发 escalate
state = {
    'conversation_id': 'test-resume',
    'messages': [{'role': 'user', 'content': '转人工', 'timestamp': datetime.now().isoformat()}],
    'routing_target': '', 'merged_response': '', 'quality_score': 0, 'quality_reason': '',
    'requires_human_review': False, 'final_response': '', 'conversation_log': [],
    'escalated': False, 'escalate_reason': '', 'handoff_context': {},
    'anomaly_counters': {}, 'sentiment': 'neutral', 'metadata': {},
}
cfg = {'configurable': {'thread_id': 'test-resume'}}
result1 = customer_service_app.invoke(state, config=cfg)

# 验证被中断
assert result1['routing_target'] == 'human_handoff'
print('Step 1: interrupted at human_handoff')

# 2. 人工确认解决 → update_state 注入人工对话上下文
# 获取中断点的 checkpoint
state_snapshot = customer_service_app.get_state(cfg)
current_state = state_snapshot.values if state_snapshot else result1
human_messages = [
    {'role': 'human_agent', 'content': '您好，我是人工客服小王，请问有什么可以帮您？'},
    {'role': 'user', 'content': '我要退款订单ORD-001'},
    {'role': 'human_agent', 'content': '好的，已为您处理退款，3个工作日内到账。'},
]
human_log = {
    'turn': 1,
    'agent': 'human',
    'user': '用户要求退款订单ORD-001',
    'reply': '人工客服处理退款，3个工作日到账',
}
customer_service_app.update_state(
    cfg,
    {
        'messages': human_messages,
        'conversation_log': [human_log],
        'escalated': False,
        'escalate_reason': '',
        'anomaly_counters': {},
        'sentiment': 'neutral',
        'merged_response': '',
    },
)
print('Step 2: update_state with human conversation')

# 3. resume
result2 = customer_service_app.invoke(None, config=cfg)
print('Step 3: resumed, final_response:', result2.get('final_response', '')[:100])
print('Scenario resume PASS: interrupt + update_state + resume works')
"
```
Expected: `Scenario resume PASS`

- [ ] **Step 5: Test normal conversation does NOT trigger handoff**

Run:
```bash
PYTHONPATH=/home/neousys/桌面/MultiAgent python3 -c "
from datetime import datetime
from core.graph import customer_service_app

state = {
    'conversation_id': 'test-normal',
    'messages': [{'role': 'user', 'content': '你们有什么产品', 'timestamp': datetime.now().isoformat()}],
    'routing_target': '', 'merged_response': '', 'quality_score': 0, 'quality_reason': '',
    'requires_human_review': False, 'final_response': '', 'conversation_log': [],
    'escalated': False, 'escalate_reason': '', 'handoff_context': {},
    'anomaly_counters': {}, 'sentiment': 'neutral', 'metadata': {},
}
cfg = {'configurable': {'thread_id': 'test-normal-2'}}
result = customer_service_app.invoke(state, config=cfg)
assert result.get('routing_target') != 'human_handoff', f'Should not route to handoff'
print('Normal flow PASS: no false trigger')
"
```
Expected: `Normal flow PASS`

- [ ] **Step 6: Run full test.py**

Run: `PYTHONPATH=/home/neousys/桌面/MultiAgent python3 test/test.py`
Expected: All 6 scenarios pass, exit code 0.
