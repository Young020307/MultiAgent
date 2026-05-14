# core/state.py
"""状态定义：Pydantic 结构化输出模型 + AgentState TypedDict。"""

import operator
from typing import Annotated, List, Literal, TypedDict
from pydantic import BaseModel, Field

class Message(TypedDict):
    role: str
    content: str
    timestamp: str


# ---- 结构化输出模型 ----

IntentType = Literal["general", "insurance_consultation", "policy_service", "claims_assistance", "renewal_addon", "escalate"]


class RouterDecision(BaseModel):
    """纯路由决策 — 只输出意图，不生成回复。"""
    intent: IntentType = Field(
        default="general",
        description="general|insurance_consultation|policy_service|claims_assistance|renewal_addon",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度 0-1")


class QualityOutput(BaseModel):
    """质量评分结构化输出。"""
    score: int = Field(ge=1, le=5, description="评分 1-5")
    reason: str = Field(default="", description="评分理由")


# ---- 日志条目类型 ----

class LogEntry(TypedDict):
    """conversation_log 的单条记录——纯自然语言，不含工具调用/JSON。"""
    turn: int
    agent: str
    user: str
    reply: str


# ---- 状态定义 ----

class AgentState(TypedDict):
    conversation_id: str
    messages: Annotated[List[Message], operator.add]
    intent: IntentType                               # router 输出的意图类型
    routing_target: str                              # 要路由到的业务 Agent
    agent_response: str                             # 业务 Agent 的回复
    quality_score: int
    quality_reason: str
    requires_human_review: bool
    final_response: str
    conversation_log: list[LogEntry]                 # 对话日志（纯自然语言摘要）
    escalated: bool
    escalate_reason: str
    handoff_context: dict
    anomaly_counters: dict
    sentiment: str
    metadata: dict
