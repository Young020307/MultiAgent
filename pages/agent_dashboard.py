#!/usr/bin/env python
"""后台客服工作台 — 处理人工交接会话。"""
import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

import streamlit as st
from datetime import datetime
from core.graph import customer_service_app
from core.handoff_registry import get_all, unregister, get as registry_get, update as registry_update

st.set_page_config(page_title="客服工作台", layout="wide")

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
    registry_update(thread_id, {"status": "active"})


def handle_send_message(thread_id: str, msg: str):
    config = {"configurable": {"thread_id": thread_id}}
    customer_service_app.update_state(config, {
        "messages": [{
            "role": "assistant",
            "content": msg,
            "timestamp": datetime.now().isoformat(),
        }],
    })
    st.session_state.agent_sent_count += 1


def handle_end_handoff(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    customer_service_app.update_state(config, {
        "messages": [{
            "role": "assistant",
            "content": "人工客服已下线",
            "timestamp": datetime.now().isoformat(),
        }],
    })
    customer_service_app.invoke(None, config=config)
    unregister(thread_id)
    st.session_state.active_thread_id = None
    st.session_state.agent_sent_count = 0


# ================================================================
# 状态 B：接管对话
# ================================================================

if active_thread_id:
    info = registry_get(active_thread_id)
    config = {"configurable": {"thread_id": active_thread_id}}

    if info is None:
        st.warning("该会话已不在待处理列表中，可能已被处理。")
        st.session_state.active_thread_id = None
        st.rerun()

    state = customer_service_app.get_state(config)
    values = state.values if state.values else {}

    st.title(f"客服工作台 — 会话 {info.get('conversation_id', '?')}")

    col1, col2 = st.columns([3, 1])
    with col1:
        reason_key = info.get("escalate_reason", "")
        st.markdown(f"**触发原因:** {REASON_LABEL.get(reason_key, reason_key)}")
        st.markdown(f"**情感检测:** {info.get('sentiment', '?')}")
    with col2:
        if st.button("结束介入", type="primary", use_container_width=True):
            handle_end_handoff(active_thread_id)
            st.rerun()

    st.divider()

    # ---- 对话记录（按时间顺序渲染） ----
    st.subheader("对话记录")
    conv_log = values.get("conversation_log", [])
    messages = values.get("messages", [])

    # 先渲染 conversation_log 中的轮次（接管前的 AI 对话）
    logged_turns = len(conv_log)
    if logged_turns > 0:
        for entry in conv_log:
            st.chat_message("user").write(entry.get("user", ""))
            st.chat_message("assistant").write(entry.get("reply", ""))

    # 再按 messages 原始顺序渲染接管后的消息
    user_idx = 0
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            user_idx += 1
            if user_idx <= logged_turns:
                continue  # 已在 conversation_log 中显示过
        st.chat_message(role).write(content)

    st.divider()

    st.subheader("发送回复")
    agent_msg = st.text_area(
        "输入回复内容",
        key=f"agent_input_{st.session_state.agent_sent_count}",
        height=80,
    )
    if st.button("发送回复", type="primary"):
        if agent_msg.strip():
            handle_send_message(active_thread_id, agent_msg.strip())
            st.rerun()
        else:
            st.warning("回复内容不能为空")

    if st.button("刷新对话"):
        st.rerun()

    st.stop()

# ================================================================
# 状态 A：待处理列表
# ================================================================

st.title("客服工作台")

pending = get_all()

if not pending:
    st.info("当前没有待处理的人工交接会话。")
    if st.button("刷新"):
        st.rerun()
    st.stop()

st.caption(f"待处理会话: {len(pending)} 条")

for thread_id, info in pending.items():
    with st.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**会话 {info.get('conversation_id', '?')}**")
            reason_key = info.get("escalate_reason", "")
            st.caption(f"触发原因: {REASON_LABEL.get(reason_key, reason_key)}")
            st.caption(f"用户消息: \"{info.get('user_msg', '')[:100]}\"")
            st.caption(f"时间: {info.get('timestamp', '?')[:19]}")
        with cols[1]:
            st.button(
                "接管",
                key=f"take_{thread_id}",
                use_container_width=True,
                on_click=handle_take_over,
                args=(thread_id,),
            )
