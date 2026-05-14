# Streamlit 用户端 & 客服端界面 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 MultiAgent 客服系统构建两个 Streamlit 页面 — 用户咨询页 + 后台客服工作台，支持人工交接流程

**Architecture:** 单进程 Streamlit app 通过 multi-page 机制（`pages/`）运行。两个页面共享 `core/graph.py` 的 `MemorySaver` 和新增的 `core/handoff_registry.py` 模块级 dict。客服端通过 `graph.update_state()` 注入人工回复，用户端通过 `graph.get_state()` 轮询新消息

**Tech Stack:** Streamlit 1.57, LangGraph (现有), Python 3.x

---

## 文件结构概览

| 文件 | 职责 |
|------|------|
| `core/handoff_registry.py` | 进程内共享 dict，记录待处理的人工交接会话 |
| `streamlit_app.py` | Streamlit 入口页 — 用户咨询聊天界面 |
| `pages/agent_dashboard.py` | 后台客服页面 — 待处理列表 + 接管对话 |

---

### Task 1: `core/handoff_registry.py` — 人工交接注册表

**职责:** 提供进程内共享的 dict 存储，让用户端注册、客服端查询/移除待处理的交接会话

**接口:** `register()`, `unregister()`, `get_all()`, `get()`

**依赖:** 无外部依赖，纯 Python 模块

---

- [ ] **Step 1: 创建 `core/handoff_registry.py`**

```python
# core/handoff_registry.py
"""进程内共享注册表 — 记录待处理的人工交接会话。

用户端（streamlit_app.py）注册一个交接后，
客服端（pages/agent_dashboard.py）可查询、接管、移除。
"""

from datetime import datetime

_registry: dict[str, dict] = {}


def register(thread_id: str, info: dict):
    """注册一个人工交接会话。

    Args:
        thread_id: LangGraph checkpointer 的 thread_id
        info: 至少包含 conversation_id, user_msg, escalate_reason, sentiment
    """
    info.setdefault("timestamp", datetime.now().isoformat())
    _registry[thread_id] = info


def unregister(thread_id: str):
    """移除一个已完成的人工交接。"""
    _registry.pop(thread_id, None)


def get_all() -> dict[str, dict]:
    """返回所有待处理的交接（新副本）。"""
    return dict(_registry)


def get(thread_id: str) -> dict | None:
    """获取单个交接信息。"""
    return _registry.get(thread_id)
```

- [ ] **Step 2: 单元测试 — 创建 `test/test_handoff_registry.py`**

```python
# test/test_handoff_registry.py
import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

from core.handoff_registry import register, unregister, get_all, get


def test_register_and_get():
    register("thread_1", {
        "conversation_id": "abc12345",
        "user_msg": "我要投诉",
        "escalate_reason": "user_requested",
        "sentiment": "critical",
    })
    info = get("thread_1")
    assert info is not None
    assert info["conversation_id"] == "abc12345"
    assert info["escalate_reason"] == "user_requested"
    assert "timestamp" in info


def test_get_all_returns_copy():
    register("thread_2", {
        "conversation_id": "def67890",
        "user_msg": "太差了",
        "escalate_reason": "negative_sentiment",
        "sentiment": "critical",
    })
    all_items = get_all()
    assert "thread_1" in all_items
    assert "thread_2" in all_items


def test_unregister():
    unregister("thread_1")
    assert get("thread_1") is None
    assert "thread_1" not in get_all()


def test_get_nonexistent():
    assert get("nonexistent") is None


def test_unregister_nonexistent_does_not_raise():
    unregister("nonexistent")  # should not raise


def test_register_timestamp_auto():
    register("thread_3", {"conversation_id": "test"})
    info = get("thread_3")
    assert "timestamp" in info
    assert info["timestamp"] != ""


def test_unregister_cleanup():
    register("thread_a", {"conversation_id": "a"})
    register("thread_b", {"conversation_id": "b"})
    unregister("thread_a")
    all_items = get_all()
    assert "thread_a" not in all_items
    assert "thread_b" in all_items
```

- [ ] **Step 3: 运行单元测试**

Run:
```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. python -m pytest test/test_handoff_registry.py -v
```
Expected: 7 tests PASS

---

### Task 2: `streamlit_app.py` — 用户咨询页面

**职责:** 用户与 AI 客服对话的主界面。正常流程显示 Agent 回复；触发人工交接后展示等待状态并轮询人工坐席回复

**关键行为:**
- 初始化 `conversation_id` / `thread_id` / `config` 到 `st.session_state`
- 用户输入 → `graph.stream(build_state(msg), config)` 执行
- 检查 `graph.get_state(config).next` 判断是否中断
- 中断态：注册到 `handoff_registry`，提供刷新按钮检查人工回复
- 人工结束后回归正常对话

**依赖:** `core.graph`, `core.handoff_registry`, `streamlit`

---

- [ ] **Step 1: 创建 `streamlit_app.py`**

```python
#!/usr/bin/env python
"""用户咨询页面 — Streamlit 入口。"""
import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

import uuid
import streamlit as st
from datetime import datetime
from core.graph import customer_service_app
from core.handoff_registry import register as registry_register, get as registry_get

st.set_page_config(page_title="AI 客服", page_icon=":speech_balloon:", layout="centered")

# ================================================================
# 初始化会话状态
# ================================================================

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())[:8]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interrupted" not in st.session_state:
    st.session_state.interrupted = False
if "processing" not in st.session_state:
    st.session_state.processing = False

conv_id = st.session_state.conversation_id
thread_id = st.session_state.thread_id
config = {"configurable": {"thread_id": thread_id}}


def build_state(message: str) -> dict:
    return {
        "conversation_id": conv_id,
        "messages": [{"role": "user", "content": message, "timestamp": datetime.now().isoformat()}],
    }


# ================================================================
# UI 头部
# ================================================================

st.title("AI 客服")
st.caption(f"会话 ID: {conv_id}")

# 对话历史渲染
for msg in st.session_state.chat_history:
    role = msg["role"]
    avatar = "user" if role == "user" else "assistant"
    with st.chat_message(avatar):
        st.write(msg["content"])

# ================================================================
# 中断等待模式 — 轮询人工坐席回复
# ================================================================

if st.session_state.interrupted:
    state = customer_service_app.get_state(config)
    all_msgs = state.values.get("messages", []) if state.values else []
    current_count = len(all_msgs)

    # 检查 graph 是否已 resume（人工交接完成）
    if not state.next:
        st.session_state.interrupted = False
        # 显示 human_handoff 节点输出的最终通知
        final_resp = state.values.get("final_response", "") if state.values else ""
        if final_resp:
            st.session_state.chat_history.append({"role": "assistant", "content": final_resp})
        st.rerun()

    # 检查 registry 是否还登记着
    if registry_get(thread_id) is None:
        st.session_state.interrupted = False
        st.rerun()

    # 渲染新出现的人工坐席消息
    new_msg_count = current_count - len(st.session_state.chat_history) - 1  # -1 for user input
    for i in range(len(st.session_state.chat_history) + 1, current_count):
        msg = all_msgs[i]
        role = "assistant" if msg.get("role") == "assistant" else "user"
        avatar = "assistant" if role == "assistant" else "user"
        with st.chat_message(avatar):
            st.write(msg.get("content", ""))
        st.session_state.chat_history.append({"role": role, "content": msg.get("content", "")})

    with st.chat_message("assistant"):
        st.info("已转接人工客服，请稍候...")

    if st.button("刷新状态"):
        st.rerun()

    st.stop()

# ================================================================
# 用户输入
# ================================================================

user_input = st.chat_input("输入您的问题...")

if user_input and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        # 执行图流程
        for event in customer_service_app.stream(
            build_state(user_input), config=config, stream_mode="updates"
        ):
            for node_name, output in event.items():
                if node_name == "format_response":
                    final = output.get("final_response", "")
                    st.session_state.chat_history.append({"role": "assistant", "content": final})

        # 检查是否在 human_handoff 前被中断
        state = customer_service_app.get_state(config)
        if state.next and "human_handoff" in state.next:
            values = state.values
            messages = values.get("messages", [])
            last_user = messages[-1]["content"] if messages else user_input

            registry_register(thread_id, {
                "conversation_id": conv_id,
                "user_msg": last_user[:200],
                "escalate_reason": values.get("escalate_reason", "unknown"),
                "sentiment": values.get("sentiment", "neutral"),
            })
            st.session_state.interrupted = True

    except Exception as e:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"系统错误: {e}",
        })

    st.session_state.processing = False
    st.rerun()
```

- [ ] **Step 2: 验证语法**

Run:
```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && python3 -c "import py_compile; py_compile.compile('streamlit_app.py', doraise=True); print('OK')"
```
Expected: OK

- [ ] **Step 3: 手动验证用户端页面**

Run:
```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && streamlit run streamlit_app.py
```

验证清单：
1. 页面正常加载，显示会话 ID
2. 输入普通问题（如"你好"）→ 显示 Agent 回复
3. 输入"我要转人工" → 显示"已转接人工客服"等待提示
4. 出现「刷新状态」按钮

---

### Task 3: `pages/agent_dashboard.py` — 后台客服页面

**职责:** 人工坐席查看待处理交接列表，接管会话，发送回复，结束介入

**两个状态:**
- 状态 A（列表）: 展示 `handoff_registry.get_all()` 的所有条目
- 状态 B（接管）: 展示单个会话详情 + 输入回复 + 结束介入

**关键行为:**
- 从 `handoff_registry` 读取待处理列表
- 「接管」→ 切换视图
- 「发送」→ `graph.update_state()` 注入人工消息
- 「结束」→ `graph.invoke(None)` resume + `unregister`

**依赖:** `core.graph`, `core.handoff_registry`, `streamlit`

---

- [ ] **Step 1: 创建 `pages/` 目录并创建 `pages/agent_dashboard.py`**

```bash
mkdir -p /home/neousys/桌面/MultiAgent/pages
```

```python
#!/usr/bin/env python
"""后台客服工作台 — 处理人工交接会话。"""
import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

import streamlit as st
from datetime import datetime
from core.graph import customer_service_app
from core.handoff_registry import get_all, unregister, get as registry_get

st.set_page_config(page_title="客服工作台", page_icon=":headphone:", layout="wide")

REASON_LABEL = {
    "user_requested": "用户主动请求转人工",
    "system_failure": "系统故障自动转人工",
    "negative_sentiment": "负面情感升级转人工",
}

# ================================================================
# 状态管理
# ================================================================

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "agent_sent_count" not in st.session_state:
    st.session_state.agent_sent_count = 0

active_thread_id = st.session_state.active_thread_id


def handle_take_over(thread_id: str):
    st.session_state.active_thread_id = thread_id
    st.session_state.agent_sent_count = 0


def handle_send_message(thread_id: str, msg: str):
    config = {"configurable": {"thread_id": thread_id}}
    customer_service_app.update_state(config, {
        "messages": [{
            "role": "assistant",
            "content": f"[人工客服] {msg}",
            "timestamp": datetime.now().isoformat(),
        }],
    })
    st.session_state.agent_sent_count += 1


def handle_end_handoff(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    customer_service_app.invoke(None, config=config)
    unregister(thread_id)
    st.session_state.active_thread_id = None
    st.session_state.agent_sent_count = 0


def handle_back_to_list():
    st.session_state.active_thread_id = None
    st.session_state.agent_sent_count = 0


# ================================================================
# 状态 B：接管对话
# ================================================================

if active_thread_id:
    info = registry_get(active_thread_id)
    config = {"configurable": {"thread_id": active_thread_id}}

    if info is None:
        st.warning("该会话已不在待处理列表中，可能已被其他坐席处理。")
        if st.button("返回列表"):
            handle_back_to_list()
            st.rerun()
        st.stop()

    # 从 graph state 读取最新消息
    state = customer_service_app.get_state(config)
    values = state.values if state.values else {}

    st.title(f"客服工作台 — 会话 {info.get('conversation_id', '?')}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**触发原因:** {REASON_LABEL.get(info.get('escalate_reason', ''), info.get('escalate_reason', '未知'))}")
        st.markdown(f"**情感检测:** {info.get('sentiment', '?')}")
    with col2:
        if st.button("返回列表", use_container_width=True):
            handle_back_to_list()
            st.rerun()
        if st.button("结束介入", type="primary", use_container_width=True):
            handle_end_handoff(active_thread_id)
            st.rerun()

    st.divider()

    # 展示对话历史
    st.subheader("对话记录")
    messages = values.get("messages", [])
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

    st.divider()

    # 坐席回复输入
    st.subheader("发送回复")
    agent_msg = st.text_area("输入回复内容", key=f"agent_input_{st.session_state.agent_sent_count}", height=80)
    if st.button("发送回复", type="primary"):
        if agent_msg.strip():
            handle_send_message(active_thread_id, agent_msg.strip())
            st.rerun()
        else:
            st.warning("回复内容不能为空")

    st.stop()

# ================================================================
# 状态 A：待处理列表
# ================================================================

st.title("客服工作台")

pending = get_all()

if not pending:
    st.info("当前没有待处理的人工交接会话。")
    st.caption("刷新页面以检查新会话")
    if st.button("手动刷新"):
        st.rerun()
    st.stop()

st.caption(f"待处理会话: {len(pending)} 条")

for thread_id, info in pending.items():
    with st.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**会话 {info.get('conversation_id', '?')}**")
            st.caption(f"触发: {REASON_LABEL.get(info.get('escalate_reason', ''), info.get('escalate_reason', '未知'))}")
            st.caption(f"用户: \"{info.get('user_msg', '')[:100]}\"")
            st.caption(f"时间: {info.get('timestamp', '?')[:19]}")
        with cols[1]:
            st.button("接管", key=f"take_{thread_id}", use_container_width=True, on_click=handle_take_over, args=(thread_id,))
```

- [ ] **Step 2: 验证语法**

Run:
```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && python3 -c "import py_compile; py_compile.compile('pages/agent_dashboard.py', doraise=True); print('OK')"
```
Expected: OK

- [ ] **Step 3: 集成验证 — 两个页面同时运行**

1. 启动用户端:
```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && streamlit run streamlit_app.py
```

2. 在浏览器中同时打开客服端:
```
http://localhost:8501/agent_dashboard
```

验证流程:
1. 用户端输入"你好" → 正常 AI 回复
2. 用户端输入"我要投诉，你们是骗子！" → 等待人工
3. 切换到客服端 → 刷新 → 看到该会话在待处理列表中
4. 点击「接管」→ 查看对话历史和触发原因
5. 输入回复并点击「发送」→ 回到用户端点击「刷新状态」→ 看到人工回复
6. 点击「结束介入」→ 回到待处理列表（空）→ 用户端恢复正常

---

## 验证注意事项

- 两个页面必须在**同一个 `streamlit run` 进程**中运行才能共享 `MemorySaver` 和 `handoff_registry`
- 用户端和客服端需要同时打开（同一浏览器不同标签页即可）
- 如果重启 Streamlit，所有状态（MemorySaver + registry）都会丢失
- `pages/agent_dashboard.py` 通过 URL `http://localhost:8501/agent_dashboard` 访问
