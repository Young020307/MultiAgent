#!/usr/bin/env python
"""用户咨询页面 — Streamlit 入口。"""
import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

import uuid
import time
import streamlit as st
from datetime import datetime
from core.graph import customer_service_app
from core.handoff_registry import register as registry_register, get as registry_get

st.set_page_config(page_title=" 咨询台 ", layout="centered")


def clean_response(text: str) -> str:
    """去掉 format_response 添加的签名块（--- 之后的内容）。"""
    idx = text.find("\n---")
    return text[:idx] if idx != -1 else text

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
if "handoff_notice_shown" not in st.session_state:
    st.session_state.handoff_notice_shown = False

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

st.title("咨询台")
st.caption(f"会话 ID: {conv_id}")

# ================================================================
# 对话历史渲染
# ================================================================

for msg in st.session_state.chat_history:
    with st.chat_message(name=msg["role"]):
        st.write(msg["content"])

# ================================================================
# 人工交接模式
# ================================================================

if st.session_state.interrupted:
    entry = registry_get(thread_id)

    # 人工坐席已结束介入 — 先捞取遗漏消息再退出
    if entry is None:
        state = customer_service_app.get_state(config)
        all_msgs = state.values.get("messages", []) if state.values else []
        shown_contents = {msg["content"] for msg in st.session_state.chat_history}
        for msg in all_msgs:
            content = msg.get("content", "")
            if content and content not in shown_contents:
                st.session_state.chat_history.append(
                    {"role": msg.get("role", "assistant"), "content": content}
                )
        st.session_state.interrupted = False
        st.session_state.handoff_notice_shown = False
        st.rerun()

    status = entry.get("status", "pending")

    # 坐席尚未接管 — 静默轮询
    if status == "pending":
        time.sleep(2)
        st.rerun()

    # 坐席已接管 — 显示通知 + 进入对话
    if status == "active":
        if not st.session_state.handoff_notice_shown:
            with st.chat_message(name="assistant"):
                st.write("已转接人工客服")
            st.session_state.chat_history.append({"role": "assistant", "content": "已转接人工客服"})
            st.session_state.handoff_notice_shown = True
            st.rerun()

        # 检查 graph state 中是否有新消息（人工坐席回复 + 用户自己发的）
        state = customer_service_app.get_state(config)
        all_msgs = state.values.get("messages", []) if state.values else []
        shown_contents = {msg["content"] for msg in st.session_state.chat_history}
        new_found = False
        for msg in all_msgs:
            content = msg.get("content", "")
            if content and content not in shown_contents:
                role = msg.get("role", "assistant")
                with st.chat_message(name=role):
                    st.write(content)
                st.session_state.chat_history.append({"role": role, "content": content})
                new_found = True

        if new_found:
            st.rerun()

        # 检查 graph 是否已 resume（坐席点了结束介入）
        if state.next is None or len(state.next) == 0:
            final_resp = state.values.get("final_response", "") if state.values else ""
            if final_resp and final_resp not in shown_contents:
                st.session_state.chat_history.append({"role": "assistant", "content": final_resp})
            st.session_state.interrupted = False
            st.session_state.handoff_notice_shown = False
            st.rerun()

        # 尚未收到坐席消息时持续轮询，收到后停止自动刷新让用户正常输入
        has_agent_reply = any(
            m["role"] == "assistant" and m["content"] != "已转接人工客服"
            for m in st.session_state.chat_history
        )
        if not has_agent_reply:
            time.sleep(2)
            st.rerun()

# ================================================================
# 用户输入
# ================================================================

placeholder = "输入消息给人工客服..." if st.session_state.interrupted else "输入您的问题..."
user_input = st.chat_input(placeholder)

if user_input:
    # 立即显示用户消息
    with st.chat_message(name="user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    if st.session_state.interrupted:
        # 人工交接模式：消息通过 update_state 发给坐席
        try:
            customer_service_app.update_state(config, {
                "messages": [{
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat(),
                }],
            })
        except Exception:
            pass
        st.rerun()

    # 正常模式：走 graph 流程
    try:
        for event in customer_service_app.stream(
            build_state(user_input), config=config, stream_mode="updates"
        ):
            for node_name, output in event.items():
                if node_name == "format_response":
                    final = output.get("final_response", "")
                    if final:
                        final_clean = clean_response(final)
                        with st.chat_message(name="assistant"):
                            st.write(final_clean)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": final_clean}
                        )

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
                "status": "pending",
            })
            st.session_state.interrupted = True
            st.session_state.handoff_notice_shown = False

    except Exception as e:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"系统错误: {e}",
        })

    st.rerun()
