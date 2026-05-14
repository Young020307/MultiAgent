# core/graph.py
"""图构建：

supervisor_router（路由 + 情感检测 + 异常计数检查）
  ├─ intent=escalate / sentiment=critical / anomaly超阈值 → human_handoff (interrupt!) → END
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
    human_handoff,
)
from core.agent_factory import wrap_agent_node
from core.logger import setup_logger

logger = setup_logger("graph")
memory = MemorySaver()

_BUSINESS_AGENTS = ("insurance_consultation", "policy_service", "claims_assistance", "renewal_addon")


# ============ 路由函数 ============

def route_after_router(state: AgentState) -> str:
    """supervisor_router 后的条件路由。"""
    if state.get("routing_target") == "human_handoff":
        logger.info("[route] → human_handoff reason=%s", state.get("escalate_reason"))
        return "human_handoff"

    target = state.get("routing_target", "conversation_agent")

    if target in _BUSINESS_AGENTS:
        logger.info("[route] intent=%s → %s", state.get("intent"), target)
        return target

    logger.info("[route] intent=%s → conversation_agent", state.get("intent"))
    return "conversation_agent"


def route_after_agent(state: AgentState) -> str:
    """业务 Agent 后：有回复走质检，无回复走兜底。"""
    return "quality_check" if state.get("agent_response") else "fallback"


def route_after_quality(state: AgentState) -> str:
    """质检后：如需人工审核 → human_review，无 → auto_approve。"""
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
customer_service_graph.add_node("human_handoff", human_handoff)

customer_service_graph.set_entry_point("supervisor_router")

# supervisor_router → conversation_agent / business / human_handoff
customer_service_graph.add_conditional_edges(
    "supervisor_router",
    route_after_router,
    {
        "conversation_agent": "conversation_agent",
        "insurance_consultation": "insurance_consultation",
        "policy_service": "policy_service",
        "claims_assistance": "claims_assistance",
        "renewal_addon": "renewal_addon",
        "human_handoff": "human_handoff",
    },
)

# conversation_agent 直接回复 → format_response
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

# fallback → format
customer_service_graph.add_edge("fallback", "format_response")

# human_handoff → END（先 interrupt 再到达）
customer_service_graph.add_edge("human_handoff", END)

customer_service_graph.add_edge("format_response", END)

customer_service_app = customer_service_graph.compile(
    checkpointer=memory,
    interrupt_before=["human_handoff"],
)
