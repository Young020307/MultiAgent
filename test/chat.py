#!/usr/bin/env python
"""交互式终端对话 — 与客服 Agent 系统实时交流，含人工交接流程。"""

import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

import uuid
from datetime import datetime
from core.graph import customer_service_app


REASON_LABEL = {
    "user_requested": "用户主动请求转人工",
    "system_failure": "系统故障自动转人工",
    "negative_sentiment": "负面情感升级转人工",
}


def human_agent_loop(app, config, state_snapshot) -> bool:
    """人工坐席交互循环。返回 True 表示人工已处理完毕可继续。"""
    values = state_snapshot.values
    reason = values.get("escalate_reason", "unknown")
    messages = values.get("messages", [])
    user_msg = messages[-1]["content"] if messages else "(无)"

    print(f"\n{'=' * 55}")
    print(f"  [人工坐席已接入]")
    print(f"  触发原因: {REASON_LABEL.get(reason, reason)}")
    print(f"  用户消息: {user_msg[:150]}")
    print(f"  ---")
    print(f"  输入回复内容后回车发送，输入 /end 结束人工介入")
    print(f"{'=' * 55}")

    human_messages = []
    while True:
        try:
            msg = input("\n[人工坐席] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n人工介入已取消，系统恢复。")
            break

        if not msg:
            continue
        if msg.lower() == "/end":
            print("人工介入结束，系统继续运行...")
            break

        human_messages.append({
            "role": "assistant",
            "content": f"[人工客服] {msg}",
            "timestamp": datetime.now().isoformat(),
        })
        print(f"  -> 已记录回复 ({len(msg)} 字符)")

    if not human_messages:
        # 人工未发送任何消息，直接 resume 让 human_handoff 执行
        for event in app.stream(None, config=config, stream_mode="updates"):
            for node_name, output in event.items():
                if node_name == "human_handoff":
                    print(f"\n[系统] {output.get('final_response', '')}")
        return True

    # 追加人工对话到 conversation_log
    old_log = values.get("conversation_log", [])
    turn = len(old_log) + 1
    new_entries = []
    for hm in human_messages:
        new_entries.append({
            "turn": turn,
            "agent": "human_agent",
            "user": user_msg,
            "reply": hm["content"],
        })
        turn += 1

    # 注入人工消息到 state，重置异常计数器
    app.update_state(config, {
        "messages": human_messages,
        "conversation_log": old_log + new_entries,
        "anomaly_counters": {},
    })

    # resume 执行 human_handoff → END
    for event in app.stream(None, config=config, stream_mode="updates"):
        for node_name, output in event.items():
            if node_name == "human_handoff":
                print(f"\n[系统] {output.get('final_response', '')}")

    return True


def main():
    conversation_id = str(uuid.uuid4())[:8]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 55)
    print("  客服 Agent 系统 — 交互模式")
    print(f"  会话 ID: {conversation_id}")
    print("  输入消息开始对话，quit/exit/q 退出，clear 清除记忆")
    print("=" * 55)

    def build_state(message: str) -> dict:
        """只传 messages，其余字段由 checkpointer 维护。"""
        return {
            "conversation_id": conversation_id,
            "messages": [{"role": "user", "content": message, "timestamp": datetime.now().isoformat()}],
        }

    while True:
        try:
            user_input = input("\n[用户] > ").strip()
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
            print("[记忆已清除，新会话开始]")
            continue

        print()
        interrupted = False

        try:
            for event in customer_service_app.stream(
                build_state(user_input), config=config, stream_mode="updates"
            ):
                for node_name, output in event.items():
                    if node_name == "supervisor_router":
                        target = output.get("routing_target", "")
                        intent = output.get("intent", "")
                        sentiment = output.get("sentiment", "")
                        extra = f" sentiment={sentiment}" if sentiment and sentiment != "neutral" else ""
                        print(f"[路由] intent={intent} -> {target}{extra}")

                    elif node_name == "conversation_agent":
                        reply = output.get("final_response", "")
                        print(f"[对话Agent] 回复 ({len(reply)} 字符)")

                    elif node_name in ("insurance_consultation", "policy_service", "claims_assistance", "renewal_addon"):
                        resp = output.get("agent_response", "")
                        print(f"  [{node_name}] 答复完成 ({len(resp)} 字符)")

                    elif node_name == "quality_check":
                        score = output["quality_score"]
                        review = "人工审核" if output.get("requires_human_review") else "通过"
                        print(f"  [质检] {score}/5 ({review})")

                    elif node_name == "human_handoff":
                        print(f"\n[系统] {output.get('final_response', '')}")

                    elif node_name == "format_response":
                        print()
                        print("-" * 40)
                        print(output.get("final_response", ""))
                        print("-" * 40)

        except Exception as e:
            print(f"[错误] {e}")
            continue

        # 检查是否在 human_handoff 前被中断
        state = customer_service_app.get_state(config)
        if state.next and "human_handoff" in state.next:
            human_agent_loop(customer_service_app, config, state)


if __name__ == "__main__":
    main()
