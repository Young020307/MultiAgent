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


def update(thread_id: str, updates: dict):
    """合并更新一个交接条目的字段。"""
    if thread_id in _registry:
        _registry[thread_id].update(updates)
