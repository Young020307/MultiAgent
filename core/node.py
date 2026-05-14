# core/node.py
"""图节点函数：supervisor_router + 合并 + 兜底 + 质检 + 审批 + 格式化 + 人工交接。"""

from datetime import datetime

from core.state import AgentState, RouterDecision, QualityOutput
from core.prompts import PROMPTS
from core.logger import setup_logger
from core.agent_factory import get_llm_for, parse_json
from core.config import CRITICAL_PATTERNS, NEGATIVE_PATTERNS, ANOMALY_THRESHOLDS
from core.tools import get_low_conf_streak

logger = setup_logger("node")


# ================================================================
# 情感检测 & 异常检查
# ================================================================

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


def _check_anomaly_thresholds(counters: dict) -> str | None:
    """检查异常计数器是否超阈值。返回触发原因或 None。"""
    for key, threshold in ANOMALY_THRESHOLDS.items():
        if counters.get(key, 0) >= threshold:
            return f"system_failure:{key}"
    return None


# ================================================================
# 路由节点
# ================================================================

def supervisor_router(state: AgentState) -> dict:
    """路由决策 + 情感检测 + 异常计数检查。"""
    messages = state["messages"]
    user_message = messages[-1]["content"]
    meta = state.get("metadata", {})

    logger.info("[router] 意图判断 conversation=%s len=%d",
                state.get("conversation_id", "?"), len(user_message))

    # 构建 conversation_log 上下文
    log = state.get("conversation_log", [])
    if log:
        lines = []
        for entry in log:
            lines.append(
                f"- [Turn {entry['turn']}] {entry['agent']}: "
                f"用户问\"{entry['user']}\" → 回复\"{entry['reply'][:80]}...\""
            )
        context = "历史轮次:\n" + "\n".join(lines)
    else:
        context = "历史轮次: （首轮对话）"

    prompt = PROMPTS["router"].format(user_message=user_message, context=context)
    raw = get_llm_for("supervisor_router").invoke(prompt).content
    decision = parse_json(raw, RouterDecision)

    intent = decision.intent

    # 情感检测
    sentiment = _detect_sentiment(user_message)

    # 异常计数器检查
    anomaly_counters = state.get("anomaly_counters", {})
    anomaly_trigger = _check_anomaly_thresholds(anomaly_counters)

    # ---- 人工交接触发优先级 ----

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

    # 场景 2：负面情感升级
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

    # 场景 3：系统故障兜底
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

    # ---- 正常业务路由 ----
    business = ["insurance_consultation", "policy_service", "claims_assistance", "renewal_addon"]

    if intent in business:
        logger.info("[router] 业务路由 intent=%s confidence=%.2f", intent, decision.confidence)
        return {
            "intent": intent,
            "routing_target": intent,
            "sentiment": sentiment,
            "metadata": {**meta, "router_intent": intent, "clarify_count": 0},
        }

    logger.info("[router] 对话路由 intent=%s", intent)
    return {
        "intent": intent,
        "routing_target": "conversation_agent",
        "sentiment": sentiment,
        "metadata": {**meta, "router_intent": intent},
    }


# ================================================================
# 人工交接节点
# ================================================================

def human_handoff(state: AgentState) -> dict:
    """打包人工交接上下文，终止自动化流程。由 interrupt_before 挂起。"""
    reason = state.get("escalate_reason", "unknown")

    handoff = {
        "reason": reason,
        "conversation_log": state.get("conversation_log", [])[-5:],
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


# ================================================================
# 兜底 & 质检 & 审批 & 格式化
# ================================================================

def fallback_reply(state: AgentState) -> dict:
    logger.warning("[fallback] conversation=%s", state.get("conversation_id", "?"))
    return {
        "final_response": "抱歉，当前系统暂时无法处理您的请求。已自动记录，客服团队将在 30 分钟内与您联系。",
        "metadata": {**state.get("metadata", {}), "fallback": True},
    }


def quality_check(state: AgentState) -> dict:
    merged = state.get("agent_response", "")
    target = state.get("routing_target", "")

    if not merged:
        return {"quality_score": 1, "requires_human_review": True, "quality_reason": "回复为空"}

    prompt = PROMPTS["quality"].format(categories=target, response=merged)
    raw = get_llm_for("quality_check").invoke(prompt).content
    output = parse_json(raw, QualityOutput)

    score = max(1, min(5, output.score))
    return {
        "quality_score": score,
        "quality_reason": output.reason,
        "requires_human_review": score < 3,
        "metadata": {**state.get("metadata", {}), "quality_checked_at": datetime.now().isoformat()},
    }


def human_review(state: AgentState) -> dict:
    return {
        "final_response": f"[已审核] {state.get('agent_response', '')}\n\n---\n*此回复已通过人工质量审核*",
        "metadata": {**state.get("metadata", {}), "human_reviewed": True, "reviewed_at": datetime.now().isoformat()},
    }


def auto_approve(state: AgentState) -> dict:
    return {
        "final_response": state.get("agent_response", ""),
        "metadata": {**state.get("metadata", {}), "auto_approved": True},
    }


def format_response(state: AgentState) -> dict:
    final = state.get("final_response", "")
    formatted = (
        f"{final}\n"
        f"---\n"
        f"**客服团队** | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"工单编号：{state.get('conversation_id', 'N/A')}"
    )

    # 追加 conversation_log
    messages = state.get("messages", [])
    user_msg = messages[-1]["content"] if messages else ""
    log = state.get("conversation_log", [])
    turn = len(log) + 1
    new_entry = {
        "turn": turn,
        "agent": state.get("routing_target", state.get("intent", "unknown")),
        "user": user_msg,
        "reply": state.get("final_response", ""),
    }

    # ---- 异常计数器管理 ----
    conv_id = state.get("conversation_id", "")
    counters = {**state.get("anomaly_counters", {})}

    # 从 tools.py 读取 low_conf 连续次数
    low_streak = get_low_conf_streak(conv_id) if conv_id else 0
    counters["low_conf_streak"] = low_streak

    # 质检失败计数
    if state.get("quality_score", 5) < 3:
        counters["quality_fail_streak"] = counters.get("quality_fail_streak", 0) + 1
    else:
        counters["quality_fail_streak"] = 0

    # 生成了有效回复 → 重置质检计数器（low_conf 由 tools.py 自行管理）
    if state.get("agent_response") and not state.get("requires_human_review", False):
        counters["quality_fail_streak"] = 0

    return {
        "final_response": formatted,
        "conversation_log": [*log, new_entry],
        "anomaly_counters": counters,
    }
