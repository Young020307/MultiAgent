# core/config.py
"""集中管理所有配置，纯数据不包含逻辑。"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ================================================================
# 通用默认值
# ================================================================

_MINIMAX_LLM = {
    "provider": "openai",
    "model": "MiniMax-M2.5-highspeed",
    "api_key_env": "MINIMAX_API_KEY",
    "api_base": "https://api.minimaxi.com/v1",
    "extra_body": {"reasoning_split": True},
}

_OLLAMA_LLM = {
    "provider": "ollama",
    "model": "qwen2.5:7b",
    "api_base": "http://127.0.0.1:11434",
}

# ================================================================
# 每个 Agent 独立的 LLM 配置
# ================================================================

AGENT_LLM_CONFIG = {
    "supervisor_router": {
        **_OLLAMA_LLM,
        "temperature": 0.1,
    },
    "conversation_agent": {
        **_OLLAMA_LLM,
        "temperature": 0.8,
    },
    "insurance_consultation": {
        **_OLLAMA_LLM,
        "temperature": 0.3,
    },
    "policy_service": {
        **_OLLAMA_LLM,
        "temperature": 0.5,
    },
    "claims_assistance": {
        **_OLLAMA_LLM,
        "temperature": 0.5,
    },
    "renewal_addon": {
        **_OLLAMA_LLM,
        "temperature": 0.5,
    },
    "merge": {
        **_OLLAMA_LLM,
        "temperature": 0.2,
    },
    "quality_check": {
        **_OLLAMA_LLM,
        "temperature": 0.1,
    },
}

# ================================================================
# 其他配置
# ================================================================

LOG_CONFIG = {
    "level": "INFO",
    "format": "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
}

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///insurance_service.db")

AGENT_TOOLS = {
    "conversation_agent": [],
    "insurance_consultation": ["search_knowledge_base"],
    "policy_service":         ["db_lookup_policy", "search_knowledge_base"],
    "claims_assistance":      ["db_lookup_claim", "db_submit_claim", "search_knowledge_base"],
    "renewal_addon":          ["db_check_renewal", "db_query_product", "search_knowledge_base"],
}

# ================================================================
# Agentic RAG 配置
# ================================================================

RELEVANCE_THRESHOLD = 0.5 

MAX_RETRIEVAL_ROUNDS = 3

REWRITE_LLM_CONFIG = {
    "provider": "ollama",
    "model": "qwen2.5:3b",
    "api_base": "http://127.0.0.1:11434",
    "temperature": 0.1,
}

# ================================================================
# 人工介入配置
# ================================================================

CRITICAL_PATTERNS = [
    "投诉", "举报", "忍无可忍", "我要告", "再也不用了",
    "骗子", "垃圾公司", "太失望了", "糟透了", "糟糕",
    "太烂了", "差劲", "太恶心了", "无法忍受", "什么垃圾",
]
NEGATIVE_PATTERNS = [
    "不满意", "太差了", "很生气", "火大", "什么玩意",
    "糊弄", "忽悠", "扯淡", "坑人",
    "不好用", "太难用了", "什么破",
]

ANOMALY_THRESHOLDS = {
    "low_conf_streak": 2,        # 连续 low_confidence 达到此值触发
    "quality_fail_streak": 1,     # quality_score < 3 出现一次即触发转人工
    "tool_error_streak": 3,       # 连续工具错误达到此值触发
}
