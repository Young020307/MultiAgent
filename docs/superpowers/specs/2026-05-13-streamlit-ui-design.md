# Streamlit 用户端 & 客服端界面 — 设计文档

> 2026-05-13 | Streamlit + LangGraph MemorySaver + handoff_registry

---

## 1. 目标

为 MultiAgent 客服系统提供两个 Streamlit 页面：
- **用户咨询页面**（入口页）：用户与 AI 客服对话，触发人工交接时自动等待
- **后台客服页面**（`pages/agent_dashboard.py`）：人工坐席查看待处理交接，接管并与用户沟通

---

## 2. 架构

```
streamlit_app.py (用户端)          pages/agent_dashboard.py (客服端)
       │                                      │
       │              ┌───────────────────────┘
       ▼              ▼
  ┌─────────────────────────┐
  │   core/handoff_registry │  共享模块，Python dict
  │   {thread_id: {         │
  │     conversation_id,    │
  │     user_msg,           │
  │     sentiment,          │
  │     escalate_reason,    │
  │     timestamp,          │
  │     agent_name,         │
  │   }}                    │
  └─────────────────────────┘
       │              │
       ▼              ▼
  ┌─────────────────────────────┐
  │  core/graph.py              │
  │  customer_service_app       │
  │  MemorySaver (进程内共享)    │
  │  interrupt_before=[         │
  │    "human_handoff"          │
  │  ]                          │
  └─────────────────────────────┘
```

**约束：** 两个页面作为同一个 Streamlit app 的 page，在同一个 Python 进程中运行，共享 `MemorySaver` 和 `handoff_registry` 模块级变量。

---

## 3. 用户端 — `streamlit_app.py`

### 3.1 状态管理（st.session_state）

| 字段 | 类型 | 用途 |
|------|------|------|
| `conversation_id` | str | 会话 ID（UUID 前 8 位） |
| `thread_id` | str | LangGraph checkpointer 的 thread_id |
| `chat_history` | list[dict] | 本地缓存的消息历史（用于快速渲染） |
| `interrupted` | bool | 当前是否处于人工交接中断状态 |
| `config` | dict | LangGraph 的 config 对象 |

### 3.2 流程

```
1. 页面初始化 → 生成 conversation_id 和 thread_id
2. 渲染 chat_history 中的所有消息
3. 用户输入 → graph.stream() 执行
4. 如果正常完成 → 显示 Agent 回复，追加到 chat_history
5. 如果中断（state.next 含 human_handoff）:
   a. 显示 "已转接人工客服，请稍候..."
   b. 注册到 handoff_registry
   c. 设置 interrupted = True
6. 如果 interrupted = True:
   a. 定时自动 rerun（st.rerun() + time.sleep(2)）
   b. 检查 graph.get_state() 是否有新消息（人工回复）
   c. 显示人工回复
   d. 检查 graph 是否已 resume 完成 → interrupted = False，回归正常
```

### 3.3 UI 布局

```
┌─────────────────────────────────────┐
│  🤖 AI 客服                         │
│  会话 ID: abc12345                  │
├─────────────────────────────────────┤
│                                     │
│  [用户] 你们的退款政策是什么？       │
│  [客服] 我们支持 7 天内无理由退款...  │
│  [用户] 我要投诉！你们太差了！       │
│  [系统] 已为您转接人工客服，请稍候... │
│  [人工客服] 您好，我是高级客服...     │
│                                     │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐    │
│  │  输入您的问题...            │    │
│  └─────────────────────────────┘    │
│  [发送]                              │
└─────────────────────────────────────┘
```

---

## 4. 客服端 — `pages/agent_dashboard.py`

### 4.1 状态管理

| 字段 | 类型 | 用途 |
|------|------|------|
| `active_thread_id` | str\|None | 当前正在处理的会话 thread_id |
| `active_context` | dict\|None | 当前会话的交接上下文 |
| `agent_messages` | list[str] | 当前坐席已发送的消息缓存 |

### 4.2 流程

```
状态 A：待处理列表
  1. 从 handoff_registry 读取所有待处理的交接
  2. 以表格/卡片形式展示
  3. 每项显示：会话 ID、用户消息摘要、触发原因、时间
  4. 点击「接管」→ 进入状态 B

状态 B：接管对话
  1. 显示用户原始消息 + conversation_log 历史
  2. 显示交接原因（负面情感 / 用户请求 / 系统故障）
  3. 输入框：坐席输入回复
  4. 点击「发送」→ graph.update_state() 注入消息
  5. 点击「结束介入」:
     a. graph.invoke(None) resume → human_handoff → END
     b. 从 handoff_registry 删除
     c. 回到状态 A
```

### 4.3 UI 布局（状态 A）

```
┌──────────────────────────────────────────────┐
│  🎧 客服工作台                                │
├──────────────────────────────────────────────┤
│  待处理会话         共 2 条                    │
│                                              │
│  ┌─────────────────────────────────────┐     │
│  │ 会话 abc123                          │     │
│  │ 触发: 负面情感升级                   │     │
│  │ 用户: "你们是骗子垃圾公司太失望了"    │     │
│  │ 时间: 2026-05-13 14:32              │     │
│  │ [接管]                               │     │
│  └─────────────────────────────────────┘     │
│                                              │
│  ┌─────────────────────────────────────┐     │
│  │ 会话 def456                          │     │
│  │ 触发: 用户请求转人工                  │     │
│  │ 用户: "我要转人工客服"                │     │
│  │ 时间: 2026-05-13 14:35              │     │
│  │ [接管]                               │     │
│  └─────────────────────────────────────┘     │
└──────────────────────────────────────────────┘
```

### 4.4 UI 布局（状态 B）

```
┌──────────────────────────────────────────────┐
│  🎧 客服工作台 — 会话 abc123                   │
│  [← 返回列表]                    [结束介入]    │
├──────────────────────────────────────────────┤
│  触发原因: 负面情感升级                       │
│  用户消息: "你们是骗子垃圾公司太失望了"        │
│  情感检测: critical                           │
│  ────────────────────────────                │
│  对话历史:                                    │
│  [用户] 你们的退款政策是什么？                │
│  [客服] 我们支持 7 天内无理由退款...           │
│  [用户] 你们是骗子垃圾公司太失望了！           │
│  ────────────────────────────                │
│  你的回复:                                    │
│  >>> "您好，我是高级客服经理..."  ✓ 已发送    │
│  >>> "非常抱歉给您带来不便..."    ✓ 已发送    │
│                                              │
│  ┌─────────────────────────────────────┐     │
│  │  输入回复...                        │     │
│  └─────────────────────────────────────┘     │
│  [发送回复]                                  │
└──────────────────────────────────────────────┘
```

---

## 5. 核心模块 — `core/handoff_registry.py`

```python
# 进程内共享注册表
# key = thread_id, value = handoff_info dict

_registry: dict[str, dict] = {}

def register(thread_id: str, info: dict):
    """注册一个人工交接会话"""

def unregister(thread_id: str):
    """移除已处理的交接"""

def get_all() -> dict[str, dict]:
    """返回所有待处理的交接"""

def get(thread_id: str) -> dict | None:
    """获取单个交接信息"""
```

---

## 6. 消息同步机制

### 6.1 客服端 → 用户端

```
agent_dashboard: graph.update_state(config, {"messages": [new_msg]})
                      │
                      ▼
              MemorySaver (进程内共享)
                      │
                      ▼
streamlit_app: graph.get_state(config) → messages 中有新消息
               → 渲染到 chat_history
```

### 6.2 接入结束通知

```
agent_dashboard: graph.invoke(None, config) → human_handoff → END
                 handoff_registry.unregister(thread_id)
                      │
                      ▼
streamlit_app: 轮询发现 interrupted=False 且 registry 中已清理
               → 显示最终回复，恢复正常输入
```

---

## 7. 文件结构

```
MultiAgent/
├── streamlit_app.py              # Streamlit 入口（用户咨询页面）
├── pages/
│   └── agent_dashboard.py        # 后台客服页面
├── core/
│   ├── handoff_registry.py       # 新增：人工交接注册表
│   ├── graph.py                  # （已有）customer_service_app
│   ├── node.py                   # （已有）节点函数
│   ├── state.py                  # （已有）AgentState
│   └── ...
└── test/
    └── chat.py                   # （已有）终端交互测试
```

---

## 8. 依赖

- `streamlit` — UI 框架
- 已有项目依赖不变

---

## 9. 限制 & 后续扩展

- **本次范围：** 一对一模式（客服一次处理一个会话），单进程运行
- **不在范围：** 多客服并发、持久化移交记录、通知推送、身份认证
- **后续可扩展：** Redis 替代进程内 dict（支持多进程）、客服登录系统、交接会话监控
