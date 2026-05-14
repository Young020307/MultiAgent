# core/agent_factory.py
"""Agent 工厂：每个 Agent 独立 LLM + create_agent 构建 + 节点包装。"""

import os
import re

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from core.config import AGENT_LLM_CONFIG, AGENT_TOOLS
from core.prompts import PROMPTS
from core.logger import setup_logger
from core.tools import KNOWLEDGE_TOOLS, BUSINESS_TOOLS

logger = setup_logger("agent_factory")

# ============ 每个 Agent 独立的 LLM 懒加载 ============

_llm_cache = {}
_agents = {}

def get_llm_for(agent_name: str):
    """获取指定 Agent 的独立 LLM 实例。

    根据 AGENT_LLM_CONFIG[agent_name]["provider"] 创建对应的客户端：
    - "openai"  → ChatOpenAI（MiniMax/OpenAI 等兼容 API）
    - "ollama"  → ChatOllama（本地模型）
    """
    if agent_name not in _llm_cache:
        cfg = AGENT_LLM_CONFIG.get(agent_name, AGENT_LLM_CONFIG["conversation_agent"])
        provider = cfg.get("provider", "openai")
        temp = cfg.get("temperature", 0.7)

        if provider == "ollama":
            _llm_cache[agent_name] = ChatOllama(
                model=cfg["model"],
                base_url=cfg.get("api_base", "http://127.0.0.1:11434"),
                temperature=temp,
            )
        else:  # openai 兼容
            _llm_cache[agent_name] = ChatOpenAI(
                model=cfg["model"],
                openai_api_key=os.getenv(cfg.get("api_key_env", "MINIMAX_API_KEY")),
                openai_api_base=cfg.get("api_base", "https://api.minimaxi.com/v1"),
                extra_body=cfg.get("extra_body"),
                temperature=temp,
            )

        logger.info("[LLM] %s -> %s:%s temperature=%.1f",
                    agent_name, provider, cfg["model"], temp)
    return _llm_cache[agent_name]


# ============ JSON 解析工具 ============

def strip_markdown_json(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    return m.group(1).strip() if m else text


def parse_json(raw: str, model_class: type):
    cleaned = strip_markdown_json(raw)
    try:
        return model_class.model_validate_json(cleaned)
    except Exception:
        logger.warning("[parse] 解析失败 raw=%s", raw[:200])
        raise


# ============ Agent 构建 ============

def _build_agent(name: str):
    """为指定类别构建 create_agent 实例，使用其专属 LLM 配置。"""
    tool_names = AGENT_TOOLS.get(name, [])
    tools = []
    for tname in tool_names:
        for tool in (KNOWLEDGE_TOOLS + BUSINESS_TOOLS.get(name, [])):
            if tool.__name__ == tname:
                tools.append(tool)
                break

    if name == "conversation_agent":
        prompt = PROMPTS["conversation_agent"]
        return create_agent(
            model=get_llm_for(name),
            tools=[],
            system_prompt=prompt,
            name=name,
            checkpointer=MemorySaver(),
        )

    prompt = PROMPTS[name].format(user_message="{user_message}", context="{context}")
    return create_agent(
        model=get_llm_for(name),
        tools=tools if tools else [],
        system_prompt=prompt,
        name=name,
        checkpointer=MemorySaver(),
    )


def get_agent(name: str):
    """懒加载获取 Agent 实例。"""
    if name not in _agents:
        _agents[name] = _build_agent(name)
    return _agents[name]


# ============ 节点包装 ============

def wrap_agent_node(agent_name: str):
    """将 create_agent 实例包装为标准 LangGraph 节点函数。"""

    def node_fn(state: AgentState) -> dict:
        agent = get_agent(agent_name)
        messages = state["messages"]
        user_message = messages[-1]["content"]
        conv_id = state.get("conversation_id", "?")

        logger.info("[%s] 激活 conversation=%s", agent_name, conv_id)

        if agent_name == "conversation_agent":
            clarify_count = state.get("metadata", {}).get("clarify_count", 0)
            chat_input = {
                "messages": [{
                    "role": "user",
                    "content": (
                        f"intent: {state.get('intent', 'greeting')}\n"
                        f"clarify_count: {clarify_count}\n"
                        f"message: {user_message}"
                    ),
                }]
            }
        else:
            chat_input = {"messages": [{"role": "user", "content": user_message}]}

        result = agent.invoke(
            chat_input,
            config={"configurable": {"thread_id": f"{conv_id}_{agent_name}"}},
        )

        response_text = ""
        for msg in reversed(result.get("messages", [])):
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            role = msg.type if hasattr(msg, "type") else msg.get("role", "")
            if content and role not in ("user", "human", ""):
                response_text = content
                break

        logger.info("[%s] 完成 conversation=%s len=%d", agent_name, conv_id, len(response_text))

        if agent_name == "conversation_agent":
            current_intent = state.get("intent", "ambiguous")
            clarify_count = state.get("metadata", {}).get("clarify_count", 0)
            new_clarify = clarify_count + 1 if current_intent == "ambiguous" else clarify_count
            return {
                "final_response": response_text,
                "metadata": {
                    **state.get("metadata", {}),
                    "clarify_count": new_clarify,
                    "replied_by": "conversation_agent",
                },
            }

        return {"agent_response": response_text}

    return node_fn
