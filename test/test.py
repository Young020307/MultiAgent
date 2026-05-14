# test.py
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
        "agent_response": "",
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
                if "agent_response" in output:
                    resp = output["agent_response"]
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
        ("投保咨询", "我想给家里人买一份重疾险，有什么推荐的吗？"),
        ("保单服务", "帮我查一下我的保单 POL-20240001，看看什么时候到期。"),
        ("理赔协助", "我的车被刮了，我已经报了案，帮我查一下理赔进度 CLM-20240001。"),
        ("续保加保", "我的保单快到期了，有什么续保优惠吗？"),
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
    stream_chat("我想了解百万医疗险的保障范围是什么？")
    print()

    # 场景 6：多轮对话（验证 conversation_log + 记忆）
    print("=" * 50)
    print("场景 6：多轮对话（验证 conversation_log + 记忆）")
    print("=" * 50)
    thread = str(uuid.uuid4())
    for i, msg in enumerate([
        "你好，我想咨询一下意外险",
        "保费大概多少钱？保障额度是多少？",
        "帮我对比一下基础版和升级版意外险",
    ], 1):
        print(f"[第{i}轮]")
        r = chat_with_customer(msg, thread_id=thread)
        print(r[:200] + "...\n")
