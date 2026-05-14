# 人工介入（Human Handoff）设计文档

> 实现三大人工触发场景：用户要求转人工、系统故障兜底、负面情感升级。

---

## 架构

```
supervisor_router (增强)
  ├─ intent=escalate ──────────────────→ human_handoff → END
  ├─ sentiment=critical ───────────────→ human_handoff → END
  ├─ anomaly_counters 超阈值 ───────────→ human_handoff → END
  └─ 正常路由（不变）
```

## 新增/修改状态字段

`AgentState` 新增：

```python
escalate_reason: str          # "" | "user_requested" | "system_failure" | "negative_sentiment"
escalated: bool               # 是否已触发人工交接
handoff_context: dict         # 交接给人工坐席的上下文包
anomaly_counters: dict        # {"low_conf_streak": 0, "tool_error_streak": 0, "quality_fail_streak": 0}
sentiment: str                # "neutral" | "negative" | "critical"
```

`IntentType` 新增 `"escalate"`。

## 三大场景 & 触发条件

### 场景 1：用户要求转人工

Router prompt 新增意图类型：
```
- escalate: 用户明确要求转人工，如"转人工"、"我要跟真人说话"、"有没有人工客服"
```

Router 识别到 `escalate` → 设置 `escalated=True`, `escalate_reason="user_requested"` → 直接路由到 `human_handoff`，完全不经过下游 Agent。

### 场景 4：系统故障/边界兜底

Supervisor 维护 `anomaly_counters`，以下条件触发：

| 计数器 | 触发条件 | 阈值 |
|--------|---------|------|
| `low_conf_streak` | 连续 `low_confidence=true` | >= 2 |
| `quality_fail_streak` | 连续 `quality_score < 3` | >= 2 |
| `tool_error_streak` | 工具调用连续异常 | >= 3 |

任一超阈值 → `escalate_reason="system_failure"` → `human_handoff`。

计数器在每次成功回复后重置。

### 场景 5：负面情感升级

轻量关键词匹配（不调 LLM，零延迟）：

```python
CRITICAL_PATTERNS = [
    "投诉", "举报", "忍无可忍", "我要告", "再也不用了",
    "骗子", "垃圾公司", "太失望了", "糟透了",
]
NEGATIVE_PATTERNS = [
    "不满意", "太差了", "很生气", "火大", "什么玩意",
    "糊弄", "忽悠", "扯淡", "坑人",
]
```

匹配到 CRITICAL → `sentiment="critical"` → `human_handoff`。
匹配到 NEGATIVE → `sentiment="negative"`，不立即升级但记录在 handoff_context 中。

## 新增节点：`human_handoff`

```python
def human_handoff(state: AgentState) -> dict:
    reason = state["escalate_reason"]
    messages = state["messages"]
    
    # 打包上下文
    handoff = {
        "reason": reason,
        "last_messages": messages[-5:],  # 最近 5 条消息
        "conversation_log": state.get("conversation_log", [])[-3:],
        "intent": state.get("intent"),
        "sentiment": state.get("sentiment", "neutral"),
        "anomaly_counters": state.get("anomaly_counters", {}),
        "timestamp": datetime.now().isoformat(),
    }
    
    # 用户提示
    user_notice = {
        "user_requested": "已为您转接人工客服，请稍候...",
        "system_failure": "系统暂时无法处理您的请求，正在为您转接人工客服...",
        "negative_sentiment": "我们非常重视您的体验，已为您转接高级人工客服...",
    }.get(reason, "正在为您转接人工客服...")
    
    return {
        "escalated": True,
        "handoff_context": handoff,
        "final_response": user_notice,
        "metadata": {**state.get("metadata", {}), "human_handoff": True},
    }
```

## 人工节点：LangGraph interrupt 真中断

`human_handoff` 不是自动继续，而是触发 LangGraph 中断：

```
supervisor_router → human_handoff (interrupt!) 
                         │
                         ├─ 1. 打包 handoff_context
                         ├─ 2. 返回转接提示给用户
                         ├─ 3. graph 挂起 (interrupt_before)
                         │
                         ═══════ ══ 人工坐席与用户多轮沟通（图外，不限轮次） ══════
                         │
                         ├─ 4. 人工坐席确认"已解决" → 调用 update_state()
                         │       ├─ messages: 追加人工期间的对话
                         │       ├─ conversation_log: 追加 {"agent": "human", ...}
                         │       ├─ escalated: False (重置)
                         │       └─ anomaly_counters: 全清零
                         │
                         └─ 5. 系统 resume → END（下一轮 supervisor 可看到人工对话上下文）
```

**图编译：** `graph.compile(checkpointer=memory, interrupt_before=["human_handoff"])`

**外部调用侧（test/chat.py）：** 捕获 `GraphInterrupt` → 展示 `handoff_context` → 等待人工输入 → `update_state` → `invoke(null)` resume。

## 人工结束后上下文注入

当人工确认解决后，`update_state` 注入以下内容到 state：

```python
{
    "messages": [
        {"role": "human_agent", "content": "人工坐席已接手..."},
        {"role": "user", "content": "用户的问题"},
        {"role": "human_agent", "content": "人工的回答"},
        # ... 多轮，由人工坐席自行记录 ...
    ],
    "conversation_log": [*old_log, {
        "turn": next_turn,
        "agent": "human",
        "user": "人工期间的对话摘要",
        "reply": "人工坐席已处理并解决",
    }],
    "escalated": False,
    "escalate_reason": "",
    "anomaly_counters": {},
    "sentiment": "neutral",
}
```

这样下一轮 supervisor 通过 `conversation_log` 和 `messages` 可完整感知人工干预期间发生了什么。

## 图变更

```
新增节点: human_handoff
新增边:
  supervisor_router → human_handoff (intent=escalate / sentiment=critical / anomaly超阈值)
  human_handoff → END

图编译: interrupt_before=["human_handoff"]
```

`route_after_router` 新增判断：
1. 先检查 `escalated` 或 `sentiment=="critical"`
2. 再检查 `anomaly_counters` 是否超阈值
3. 最后走原有业务路由

异常计数器更新逻辑：在 `format_response` 节点中，根据上一轮结果更新 `anomaly_counters`（成功则清零对应项，失败则累加）。

## 文件变更

| 文件 | 变更 |
|------|------|
| `core/state.py` | 新增 `escalate_reason`、`escalated`、`handoff_context`、`anomaly_counters`、`sentiment` 字段；`IntentType` 新增 `"escalate"` |
| `core/prompts.py` | Router prompt 新增 escalate 意图说明 |
| `core/node.py` | 新增 `human_handoff` 节点；`supervisor_router` 新增情感检测和异常计数检查 |
| `core/graph.py` | 新增 `human_handoff` 节点和路由边；编译时 `interrupt_before=["human_handoff"]` |
| `core/config.py` | 新增 `CRITICAL_PATTERNS`、`NEGATIVE_PATTERNS`、`ANOMALY_THRESHOLDS` |
| `test/test.py` | 新增人工交接场景测试；新增 interrupt + resume 交互用例 |

## 测试要点

1. 用户输入"转人工" → router 识别 escalate → human_handoff → graph 挂起
2. 连续 2 次 low_confidence → anomaly_counters 超阈值 → human_handoff
3. 用户输入"垃圾公司忍无可忍" → sentiment=critical → human_handoff
4. 正常对话不触发误升级
5. 人工确认后 update_state → resume → 下一轮 supervisor 能感知人工对话
6. `test.py` 无回归
