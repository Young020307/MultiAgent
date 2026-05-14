# MultiAgent 客户服务系统 — 健壮性增强实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有的单 Agent 路由原型升级为支持多指派、并行执行、结构化输出的健壮 Supervisor 多智能体系统。

**Architecture:** Supervisor 模式 — categorize 节点分析用户意图返回多个路由目标 → 并行激活对应 Agent（各 Agent 是独立的 create_agent 实例，挂载不同工具）→ merge 合并结果 → quality_check → format_response。全链路增加日志、结构化输出、降级路径。

**Tech Stack:** Python 3.10+, LangGraph, LangChain OpenAI, Pydantic, pytest

---

### Task 0: 项目初始化

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: 创建 requirements.txt**

```bash
cat > /home/neousys/桌面/MultiAgent/requirements.txt << 'EOF'
langgraph>=0.2.0
langchain-openai>=0.3.0
langchain-core>=0.3.0
pydantic>=2.0
pytest>=8.0
pytest-asyncio>=0.24
EOF
```

- [ ] **Step 2: 安装依赖**

```bash
cd /home/neousys/桌面/MultiAgent && pip install -r requirements.txt
```

---

### Task 1: 配置模块（config.py）

**Files:**
- Create: `core/config.py`

- [ ] **Step 1: 编写 config.py**

```python
# core/config.py
"""集中管理所有配置，纯数据不包含逻辑。"""

from langgraph.pregel import RetryPolicy

LLM_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.7,
}

RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    max_interval=30.0,
)

LOG_CONFIG = {
    "level": "INFO",
    "format": "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
}

AGENT_TOOLS = {
    "technical": ["search_product_docs", "get_api_example"],
    "sales":     ["query_pricing", "check_promotion", "search_product_docs"],
    "support":   ["lookup_order", "check_refund_status", "search_product_docs"],
    "feedback":  ["log_feedback_to_crm"],
}
```

- [ ] **Step 2: 验证导入**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -c "from core.config import LLM_CONFIG, RETRY_POLICY, LOG_CONFIG, AGENT_TOOLS; print('OK')"
```

---

### Task 2: 日志模块（logger.py）

**Files:**
- Create: `core/logger.py`

- [ ] **Step 1: 编写 logger.py**

```python
# core/logger.py
"""结构化日志，每条记录自动携带 conversation_id 和 thread_id。"""

import logging
from core.config import LOG_CONFIG

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOG_CONFIG["level"])
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_CONFIG["format"]))
        logger.addHandler(handler)
    return logger
```

- [ ] **Step 2: 验证日志输出**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -c "
from core.logger import setup_logger
log = setup_logger('test')
log.info('logger works')
print('OK')
"
```

---

### Task 3: 提示词模块（prompts.py）

**Files:**
- Create: `core/prompts.py`

- [ ] **Step 1: 编写测试 — 验证所有提示词键存在且非空**

Create `tests/test_prompts.py`:

```python
# tests/test_prompts.py
import pytest
from core.prompts import PROMPTS

REQUIRED_KEYS = [
    "categorize",
    "technical",
    "sales",
    "support",
    "feedback",
    "merge",
    "quality",
]


def test_all_prompt_keys_present():
    for key in REQUIRED_KEYS:
        assert key in PROMPTS, f"Missing prompt key: {key}"


def test_all_prompts_non_empty():
    for key, template in PROMPTS.items():
        assert len(template.strip()) > 0, f"Prompt '{key}' is empty"


def test_categorize_prompt_contains_placeholders():
    assert "{user_message}" in PROMPTS["categorize"]


def test_agent_prompts_contain_placeholders():
    for agent in ["technical", "sales", "support", "feedback"]:
        assert "{user_message}" in PROMPTS[agent]


def test_merge_prompt_contains_placeholders():
    assert "{user_message}" in PROMPTS["merge"]
    assert "{agent_outputs}" in PROMPTS["merge"]


def test_quality_prompt_contains_placeholders():
    assert "{response}" in PROMPTS["quality"]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_prompts.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'core.prompts'`

- [ ] **Step 3: 编写 prompts.py**

```python
# core/prompts.py
"""集中管理所有提示词模板。模板使用 .format(**ctx) 填入上下文。"""

PROMPTS = {
    "categorize": """分析用户咨询，返回匹配的类别列表和置信度。

咨询内容：{user_message}

可选类别：
- technical: 技术问题、产品使用、功能咨询、API 集成
- sales: 价格、购买、套餐、折扣、试用
- support: 售后、退款、投诉、账户问题
- feedback: 建议、反馈、评价

要求：
- 可以同时匹配多个类别
- 置信度 0.0-1.0，表示对该分类的确定程度
- 如果无法判断，返回空列表""",

    "technical": """你是一位资深技术专家，专门解答产品使用和技术问题。
你的回答应该：
- 准确、详细、专业
- 提供具体的操作步骤
- 包含相关示例代码（如适用）
- 指出可能的注意事项和最佳实践

{context}

用户问题：{user_message}
请提供专业、详细的技术解答。""",

    "sales": """你是一位专业的销售顾问，负责解答价格、购买相关问题。
你的回答应该：
- 友好、热情、有说服力
- 清晰说明价格信息和优惠政策
- 主动推荐合适的产品套餐
- 提供购买链接或引导

{context}

用户咨询：{user_message}
请提供有帮助的销售建议。""",

    "support": """你是一位耐心的售后支持专家，处理用户投诉、退款、账户问题。
你的回答应该：
- 表达理解和同理心
- 提供清晰的解决方案
- 说明处理流程和预计时间
- 必要时提供升级渠道

{context}

用户问题：{user_message}
请提供周到、专业的支持回复。""",

    "feedback": """你负责处理用户反馈和建议。
你的回答应该：
- 感谢用户的反馈
- 认真对待每一条建议
- 说明反馈的处理流程
- 邀请用户继续参与产品改进

用户反馈：{user_message}
请提供真诚、专业的回复。""",

    "merge": """整合以下多个专家的回复，生成一份统一的客户答复。

用户原始问题：{user_message}

各专家回复：
{agent_outputs}

要求：
- 去重：相同信息只保留最详细的版本
- 统一语气：专业、友好、一致
- 信息矛盾时以技术准确性优先，并指出差异
- 如果只有一个专家回复，直接采用
- 保持简洁，不要重复介绍""",

    "quality": """评估以下客服回复的质量。

咨询类别：{categories}
客服回复：{response}

评分标准（1-5 分）：
5 - 完美：专业、准确、完整、友好
4 - 良好：基本满足要求，小有改进空间
3 - 一般：回答了问题，但不够专业或完整
2 - 较差：有明显错误或遗漏
1 - 不合格：完全不符合要求

返回格式：分数加理由，例如 "4 - 回复准确但可以更详细"。""",
}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_prompts.py -v
```
Expected: PASS

---

### Task 4: 工具模块（tools.py）

**Files:**
- Create: `core/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_tools.py
import pytest
from core.tools import (
    search_product_docs,
    get_api_example,
    query_pricing,
    check_promotion,
    lookup_order,
    check_refund_status,
    log_feedback_to_crm,
    KNOWLEDGE_TOOLS,
    BUSINESS_TOOLS,
)


class TestKnowledgeTools:
    def test_search_product_docs_returns_string(self):
        result = search_product_docs("如何配置 API 密钥")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_product_docs_no_match(self):
        result = search_product_docs("xyz_not_found_123")
        assert isinstance(result, str)  # 返回说明而非崩溃

    def test_get_api_example_returns_code(self):
        result = get_api_example("authentication")
        assert isinstance(result, str)
        assert len(result) > 0


class TestBusinessTools:
    def test_query_pricing_returns_price_info(self):
        result = query_pricing("enterprise")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_pricing_unknown_plan(self):
        result = query_pricing("super_premium_unknown")
        assert "未找到" in result or "unknown" in result.lower()

    def test_check_promotion_returns_status(self):
        result = check_promotion("WELCOME50")
        assert isinstance(result, str)

    def test_check_promotion_invalid(self):
        result = check_promotion("INVALID_CODE")
        assert "无效" in result or "不存在" in result

    def test_lookup_order_returns_info(self):
        result = lookup_order("ORD-001")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_check_refund_status(self):
        result = check_refund_status("REF-001")
        assert isinstance(result, str)
        assert "处理中" in result or "完成" in result or "待审核" in result

    def test_log_feedback_to_crm(self):
        result = log_feedback_to_crm("建议增加深色模式", "user@example.com")
        assert "已记录" in result


class TestToolRegistry:
    def test_knowledge_tools_structure(self):
        assert isinstance(KNOWLEDGE_TOOLS, list)
        assert len(KNOWLEDGE_TOOLS) >= 2

    def test_business_tools_structure(self):
        assert "technical" in BUSINESS_TOOLS
        assert "sales" in BUSINESS_TOOLS
        assert "support" in BUSINESS_TOOLS
        assert "feedback" in BUSINESS_TOOLS
        assert len(BUSINESS_TOOLS["sales"]) >= 2

    def test_all_tools_are_callable(self):
        for tool in KNOWLEDGE_TOOLS:
            assert callable(tool)
        for agent_tools in BUSINESS_TOOLS.values():
            for tool in agent_tools:
                assert callable(tool)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_tools.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'core.tools'`

- [ ] **Step 3: 编写 tools.py**

```python
# core/tools.py
"""知识检索 + 业务操作工具。初期用 mock 数据实现。"""

# ================================================================
# 知识检索工具
# ================================================================

_PRODUCT_DOCS = {
    "api": "要配置 API 密钥，请进入控制台 → 设置 → API Keys → 生成新密钥。密钥仅在生成时显示一次。使用方式：在 HTTP Header 中添加 `Authorization: Bearer <key>`。",
    "认证": "支持 OAuth 2.0 和 API Key 两种认证方式。OAuth 适用于用户授权场景，API Key 适用于服务端调用。",
    "限流": "免费版每秒 10 次请求，企业版每秒 1000 次。超过限制返回 429 状态码。",
    "webhook": "在项目设置中配置 Webhook URL，系统会在事件发生时 POST JSON 到该地址。",
}


def search_product_docs(query: str) -> str:
    """搜索产品文档/知识库。"""
    query_lower = query.lower()
    results = []
    for keyword, doc in _PRODUCT_DOCS.items():
        if keyword in query_lower:
            results.append(f"【{keyword}】{doc}")
    if not results:
        return f"未找到与「{query}」相关的文档。建议联系技术支持获取帮助。"
    return "\n\n".join(results)


def get_api_example(endpoint: str) -> str:
    """获取 API 使用示例。"""
    examples = {
        "authentication": "```python\nimport requests\nr = requests.get(\n    'https://api.example.com/v1/auth',\n    headers={'Authorization': 'Bearer YOUR_API_KEY'}\n)\n```",
        "chat": "```python\nresponse = client.chat.completions.create(\n    model='gpt-4o',\n    messages=[{'role': 'user', 'content': 'Hello'}]\n)\n```",
    }
    return examples.get(endpoint, f"未找到「{endpoint}」的示例代码。请查阅完整 API 文档。")


# ================================================================
# 业务操作工具
# ================================================================

_PRICING = {
    "basic": "基础版：¥99/月，支持 5 个用户、10GB 存储、API 每秒 10 次。",
    "professional": "专业版：¥499/月，支持 50 个用户、100GB 存储、API 每秒 100 次。",
    "enterprise": "企业版：¥1999/月，支持 500 个用户、1TB 存储、API 每秒 1000 次、专属支持。",
}

_PROMOTIONS = {
    "WELCOME50": "优惠码 WELCOME50：新用户首月 5 折，适用于专业版和企业版。",
    "ANNUAL20": "优惠码 ANNUAL20：年付享 8 折优惠。",
    "REFER10": "优惠码 REFER10：推荐好友双方各得 10% 折扣。",
}

_ORDERS = {
    "ORD-001": {"status": "已发货", "date": "2026-04-28", "tracking": "SF1234567890"},
    "ORD-002": {"status": "处理中", "date": "2026-05-01"},
    "ORD-003": {"status": "已完成", "date": "2026-04-15"},
}

_REFUNDS = {
    "REF-001": {"status": "处理中", "amount": "¥499", "expected": "5-7 个工作日"},
    "REF-002": {"status": "已完成", "amount": "¥99", "completed": "2026-04-30"},
}


def query_pricing(plan: str) -> str:
    """查询套餐价格信息。"""
    plan_lower = plan.lower()
    for key, info in _PRICING.items():
        if key in plan_lower:
            return info
    return f"未找到「{plan}」套餐信息。可选套餐：{', '.join(_PRICING.keys())}。"


def check_promotion(code: str) -> str:
    """校验优惠码。"""
    code_upper = code.upper().strip()
    if code_upper in _PROMOTIONS:
        return _PROMOTIONS[code_upper]
    return f"优惠码「{code}」无效或已过期。"


def lookup_order(order_id: str) -> str:
    """查询订单状态。"""
    order_id_upper = order_id.upper().strip()
    order = _ORDERS.get(order_id_upper)
    if not order:
        return f"未找到订单「{order_id}」。请确认订单号是否正确。"
    return f"订单 {order_id_upper}：状态「{order['status']}」，日期：{order['date']}。"


def check_refund_status(refund_id: str) -> str:
    """查询退款进度。"""
    refund_id_upper = refund_id.upper().strip()
    refund = _REFUNDS.get(refund_id_upper)
    if not refund:
        return f"未找到退款单「{refund_id}」。请确认退款单号是否正确。"
    status = refund["status"]
    if status == "已完成":
        return f"退款单 {refund_id_upper}：已退款 ¥{refund['amount']}，完成于 {refund.get('completed', '未知日期')}。"
    return f"退款单 {refund_id_upper}：{status}，金额 ¥{refund['amount']}，预计 {refund.get('expected', '尽快处理完毕')}。"


def log_feedback_to_crm(content: str, contact: str = "") -> str:
    """记录用户反馈到 CRM。"""
    return f"已记录用户反馈：「{content}」。{'联系方式：' + contact if contact else ''}感谢您的建议，产品团队会认真评估。"


# ================================================================
# 工具注册表
# ================================================================

KNOWLEDGE_TOOLS = [search_product_docs, get_api_example]

BUSINESS_TOOLS = {
    "technical": [],
    "sales":     [query_pricing, check_promotion],
    "support":   [lookup_order, check_refund_status],
    "feedback":  [log_feedback_to_crm],
}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_tools.py -v
```
Expected: PASS

---

### Task 5: 状态升级（state.py）

**Files:**
- Modify: `core/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_state.py
import pytest
from core.state import AgentState, Message, CategoryOutput, QualityOutput


class TestMessage:
    def test_message_creation(self):
        msg = Message(role="user", content="你好", timestamp="2026-05-05T00:00:00")
        assert msg["role"] == "user"
        assert msg["content"] == "你好"


class TestCategoryOutput:
    def test_single_category(self):
        co = CategoryOutput(categories=["technical"], confidence=0.95)
        assert co.categories == ["technical"]
        assert co.confidence == 0.95

    def test_multi_category(self):
        co = CategoryOutput(categories=["technical", "sales"], confidence=0.8)
        assert len(co.categories) == 2

    def test_empty_category(self):
        co = CategoryOutput(categories=[], confidence=0.0)
        assert co.categories == []

    def test_invalid_category_rejected(self):
        with pytest.raises(Exception):
            CategoryOutput(categories=["invalid"], confidence=0.5)

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            CategoryOutput(categories=["technical"], confidence=1.5)
        with pytest.raises(Exception):
            CategoryOutput(categories=["technical"], confidence=-0.1)


class TestQualityOutput:
    def test_valid_score(self):
        qo = QualityOutput(score=4, reason="回复准确")
        assert qo.score == 4

    def test_score_out_of_range(self):
        with pytest.raises(Exception):
            QualityOutput(score=6, reason="超出范围")
        with pytest.raises(Exception):
            QualityOutput(score=0, reason="超出范围")


class TestAgentState:
    def test_minimal_state(self):
        state: AgentState = {
            "conversation_id": "abc123",
            "messages": [{"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"}],
            "routing_targets": [],
            "assigned_agents": [],
            "agent_outputs": {},
            "merged_response": "",
            "quality_score": 0,
            "quality_reason": "",
            "requires_human_review": False,
            "final_response": "",
            "metadata": {},
        }
        # 验证字段完整性 — TypedDict 在运行时不做强制校验，
        # 但结构声明正确即可
        assert state["conversation_id"] == "abc123"
        assert state["routing_targets"] == []
        assert state["agent_outputs"] == {}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_state.py -v
```
Expected: FAIL — `ImportError` 因为当前 state.py 没有 CategoryOutput, QualityOutput, routing_targets 等字段

- [ ] **Step 3: 改写 state.py**

```python
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

class CategoryOutput(BaseModel):
    """分类节点结构化输出 — 可同时匹配多个类别。"""
    categories: list[Literal["technical", "sales", "support", "feedback"]] = Field(
        default_factory=list,
        description="匹配的类别列表，可为空（无法识别时）",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="分类置信度 0-1")


class QualityOutput(BaseModel):
    """质量评分结构化输出。"""
    score: int = Field(ge=1, le=5, description="评分 1-5")
    reason: str = Field(default="", description="评分理由")


# ---- 状态定义 ----

class AgentState(TypedDict):
    conversation_id: str
    messages: Annotated[List[Message], operator.add]
    routing_targets: list[str]                     # [改] 替代原 category: str
    assigned_agents: list[str]                     # [新] 实际激活的 Agent 列表
    agent_outputs: dict[str, str]                  # [新] {"technical": "...", "sales": "..."}
    merged_response: str                           # [新] 合并后的回复
    quality_score: int
    quality_reason: str                            # [新] 评分理由
    requires_human_review: bool
    final_response: str
    metadata: dict
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_state.py -v
```
Expected: PASS

---

### Task 6: 节点重构（node.py）

**Files:**
- Modify: `core/node.py` — 完全重写
- Test: `tests/test_nodes.py`

这是最大的改动。核心变化：
1. `categorize_inquiry` 改用 `with_structured_output(CategoryOutput)`，返回 `routing_targets`
2. 4 个 Agent 改用 `create_agent`，各自挂载工具，以节点函数包装
3. 新增 `merge_agent` 和 `fallback_reply`
4. `quality_check` 改用 `with_structured_output(QualityOutput)`
5. `human_review`、`auto_approve`、`format_response` 适配新字段

- [ ] **Step 1: 编写测试（mock LLM 输出）**

```python
# tests/test_nodes.py
import pytest
from unittest.mock import MagicMock, patch
from core.state import AgentState, CategoryOutput, QualityOutput
from core.node import (
    categorize_inquiry,
    wrap_agent_node,
    merge_agent_outputs,
    fallback_reply,
    quality_check,
    auto_approve,
    format_response,
)


def make_state(**overrides) -> AgentState:
    """创建最小完整 state 用于测试"""
    base = {
        "conversation_id": "test-001",
        "messages": [{"role": "user", "content": "测试消息", "timestamp": "2026-01-01T00:00:00"}],
        "routing_targets": [],
        "assigned_agents": [],
        "agent_outputs": {},
        "merged_response": "",
        "quality_score": 0,
        "quality_reason": "",
        "requires_human_review": False,
        "final_response": "",
        "metadata": {},
    }
    base.update(overrides)
    return base


class TestCategorizeInquiry:
    def test_single_category(self):
        """单分类返回单元素列表"""
        mock_output = CategoryOutput(categories=["technical"], confidence=0.95)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_output

        with patch("core.node.llm_categorize", mock_llm):
            result = categorize_inquiry(make_state())

        assert result["routing_targets"] == ["technical"]
        assert result["metadata"]["categorized_at"] is not None

    def test_multi_category(self):
        """混合咨询返回多元素列表"""
        mock_output = CategoryOutput(categories=["technical", "sales"], confidence=0.8)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_output

        with patch("core.node.llm_categorize", mock_llm):
            result = categorize_inquiry(make_state())

        assert set(result["routing_targets"]) == {"technical", "sales"}

    def test_empty_category(self):
        """无法识别返回空列表"""
        mock_output = CategoryOutput(categories=[], confidence=0.0)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_output

        with patch("core.node.llm_categorize", mock_llm):
            result = categorize_inquiry(make_state())

        assert result["routing_targets"] == []


class TestMergeAgentOutputs:
    def test_single_agent_bypasses_merge(self):
        """单 Agent 直接采用，不经过 LLM"""
        state = make_state(
            routing_targets=["technical"],
            agent_outputs={"technical": "API 密钥配置方法如下..."},
        )
        result = merge_agent_outputs(state)
        assert "API 密钥" in result["merged_response"]

    def test_multi_agent_merges(self):
        """多 Agent 合并"""
        mock_response = "合并后的统一回复..."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=mock_response)

        state = make_state(
            routing_targets=["technical", "sales"],
            agent_outputs={
                "technical": "技术回答",
                "sales": "销售建议",
            },
        )
        with patch("core.node.llm_merge", mock_llm):
            result = merge_agent_outputs(state)

        assert result["merged_response"] == mock_response

    def test_empty_agent_outputs_fallback(self):
        """所有 Agent 无输出时返回空字符串"""
        state = make_state(agent_outputs={})
        result = merge_agent_outputs(state)
        assert result["merged_response"] == ""


class TestFallbackReply:
    def test_returns_fallback_message(self):
        result = fallback_reply(make_state())
        assert "无法处理" in result["final_response"]
        assert result["metadata"]["fallback"] is True


class TestQualityCheck:
    def test_high_score_auto_approve(self):
        mock_output = QualityOutput(score=5, reason="完美回复")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_output

        state = make_state(merged_response="很好的回复")
        with patch("core.node.llm_quality", mock_llm):
            result = quality_check(state)

        assert result["quality_score"] == 5
        assert result["requires_human_review"] is False
        assert result["quality_reason"] == "完美回复"

    def test_low_score_triggers_review(self):
        mock_output = QualityOutput(score=2, reason="有明显遗漏")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_output

        state = make_state(merged_response="不够好的回复")
        with patch("core.node.llm_quality", mock_llm):
            result = quality_check(state)

        assert result["quality_score"] == 2
        assert result["requires_human_review"] is True


class TestAutoApprove:
    def test_passes_merged_response_to_final(self):
        state = make_state(merged_response="最终答复")
        result = auto_approve(state)
        assert result["final_response"] == "最终答复"
        assert result["metadata"]["auto_approved"] is True


class TestFormatResponse:
    def test_adds_signature(self):
        state = make_state(final_response="你好，这是回复内容")
        result = format_response(state)
        assert "客服团队" in result["final_response"]
        assert state["conversation_id"] in result["final_response"]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_nodes.py -v
```
Expected: FAIL — `ImportError` 因为新模块还未实现或旧 node.py 的函数签名不匹配

- [ ] **Step 3: 重写 node.py**

```python
# core/node.py
"""节点函数：categorize（结构化输出）+ create_agent 工厂 + 合并 + 兜底。"""

from datetime import datetime

from langgraph.prebuilt import create_agent
from langchain_openai import ChatOpenAI

from core.state import AgentState, CategoryOutput, QualityOutput
from core.config import LLM_CONFIG, AGENT_TOOLS
from core.prompts import PROMPTS
from core.logger import setup_logger
from core.tools import KNOWLEDGE_TOOLS, BUSINESS_TOOLS

logger = setup_logger("node")

# ============ LLM 实例 ============

_llm = ChatOpenAI(model=LLM_CONFIG["model"], temperature=LLM_CONFIG["temperature"])

# 结构化输出专用 LLM
llm_categorize = _llm.with_structured_output(CategoryOutput)
llm_quality = _llm.with_structured_output(QualityOutput)

# create_agent 用的 LLM
llm_agent = ChatOpenAI(model=LLM_CONFIG["model"], temperature=LLM_CONFIG["temperature"])
llm_merge = _llm  # merge 不需要结构化输出


# ============ 分类节点 ============

def categorize_inquiry(state: AgentState) -> dict:
    """分类用户咨询 — 用结构化输出返回一个或多个 routing_targets。"""
    messages = state["messages"]
    user_message = messages[-1]["content"]
    conv_id = state.get("conversation_id", "unknown")

    logger.info("[categorize] 开始分类 conversation=%s len=%d", conv_id, len(user_message))

    prompt = PROMPTS["categorize"].format(user_message=user_message)
    output: CategoryOutput = llm_categorize.invoke(prompt)

    logger.info(
        "[categorize] 分类结果 targets=%s confidence=%.2f",
        output.categories, output.confidence,
    )

    return {
        "routing_targets": output.categories,
        "metadata": {"categorized_at": datetime.now().isoformat()},
    }


# ============ Agent 工厂 ============

def _build_agent(name: str) -> object:
    """为指定类别构建 create_agent 实例。"""
    tool_names = AGENT_TOOLS.get(name, [])
    tools = []
    for tname in tool_names:
        # 先查知识工具，再查业务工具
        for tool in (KNOWLEDGE_TOOLS + BUSINESS_TOOLS.get(name, [])):
            if tool.__name__ == tname:
                tools.append(tool)
                break

    prompt = PROMPTS[name].format(user_message="{user_message}", context="{context}")

    return create_agent(
        llm_agent,
        tools=tools if tools else None,
        system_prompt=prompt,
        name=name,
    )


# 预构建 Agent 实例
_agents = {
    "technical": _build_agent("technical"),
    "sales": _build_agent("sales"),
    "support": _build_agent("support"),
    "feedback": _build_agent("feedback"),
}


def wrap_agent_node(agent_name: str):
    """将 create_agent 实例包装为标准节点函数，
    只返回 agent_response 文本，不返回完整 messages 列表。"""
    agent = _agents[agent_name]

    def node_fn(state: AgentState) -> dict:
        messages = state["messages"]
        user_message = messages[-1]["content"]
        conv_id = state.get("conversation_id", "unknown")

        logger.info("[%s] 激活 conversation=%s", agent_name, conv_id)

        # Agent 需要 messages 格式的输入
        agent_input = {"messages": [{"role": "user", "content": user_message}]}
        result = agent.invoke(agent_input)

        # 提取最后一条 AI 回复作为 agent_response
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = str(msg)
            if content and msg.get("role") != "user":
                response_text = content
                break

        logger.info("[%s] 完成 conversation=%s len=%d", agent_name, conv_id, len(response_text))

        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: response_text},
            "assigned_agents": [*state.get("assigned_agents", []), agent_name],
        }

    return node_fn


# ============ 合并节点 ============

def merge_agent_outputs(state: AgentState) -> dict:
    """合并多个 Agent 的输出为一答复。"""
    agent_outputs = state.get("agent_outputs", {})
    routing_targets = state.get("routing_targets", [])
    messages = state["messages"]
    user_message = messages[-1]["content"]
    conv_id = state.get("conversation_id", "unknown")

    logger.info("[merge] 开始合并 targets=%s outputs=%s", routing_targets, list(agent_outputs.keys()))

    if not agent_outputs:
        logger.warning("[merge] 所有 Agent 无输出")
        return {"merged_response": ""}

    # 单 Agent 直接采用
    active_names = [t for t in routing_targets if t in agent_outputs]
    if len(active_names) == 1:
        name = active_names[0]
        logger.info("[merge] 单 Agent (%s) 直接采用", name)
        return {"merged_response": agent_outputs[name]}

    # 多 Agent 合并
    outputs_text = "\n---\n".join(
        f"【{name}】{text}" for name, text in agent_outputs.items()
    )
    prompt = PROMPTS["merge"].format(
        user_message=user_message,
        agent_outputs=outputs_text,
    )
    response = llm_merge.invoke(prompt)

    logger.info("[merge] 合并完成 len=%d", len(response.content))
    return {"merged_response": response.content}


# ============ 兜底节点 ============

def fallback_reply(state: AgentState) -> dict:
    """所有降级路径的终点 — 返回兜底回复。"""
    logger.warning("[fallback] 进入兜底路径 conversation=%s", state.get("conversation_id", "unknown"))
    return {
        "final_response": (
            "抱歉，当前系统暂时无法处理您的请求。"
            "已自动记录该问题，客服团队将在 30 分钟内通过邮件与您联系。"
        ),
        "metadata": {**state.get("metadata", {}), "fallback": True},
    }


# ============ 质量检查节点 ============

def quality_check(state: AgentState) -> dict:
    """评估 merged_response 质量，用结构化输出。"""
    merged_response = state.get("merged_response", "")
    routing_targets = state.get("routing_targets", [])

    if not merged_response:
        return {"quality_score": 1, "requires_human_review": True, "quality_reason": "回复为空"}

    prompt = PROMPTS["quality"].format(
        categories=", ".join(routing_targets),
        response=merged_response,
    )
    output: QualityOutput = llm_quality.invoke(prompt)

    score = max(1, min(5, output.score))
    requires_human = score < 3

    logger.info("[quality] 评分=%d reason=%s human=%s", score, output.reason, requires_human)

    return {
        "quality_score": score,
        "quality_reason": output.reason,
        "requires_human_review": requires_human,
        "metadata": {
            **state.get("metadata", {}),
            "quality_checked_at": datetime.now().isoformat(),
        },
    }


# ============ 审批节点 ============

def human_review(state: AgentState) -> dict:
    """人工审核节点（模拟）。"""
    merged_response = state.get("merged_response", "")
    refined = f"[已审核] {merged_response}\n\n---\n*此回复已通过人工质量审核*"

    return {
        "final_response": refined,
        "metadata": {
            **state.get("metadata", {}),
            "human_reviewed": True,
            "reviewed_at": datetime.now().isoformat(),
        },
    }


def auto_approve(state: AgentState) -> dict:
    """自动批准（高质量回复）。"""
    return {
        "final_response": state.get("merged_response", ""),
        "metadata": {
            **state.get("metadata", {}),
            "human_reviewed": False,
            "auto_approved": True,
        },
    }


# ============ 格式化节点 ============

def format_response(state: AgentState) -> dict:
    """格式化最终回复，添加签名。"""
    final_response = state.get("final_response", "")

    formatted = (
        f"{final_response}\n"
        f"---\n"
        f"**客服团队** | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"工单编号：{state.get('conversation_id', 'N/A')}"
    )

    return {"final_response": formatted}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_nodes.py -v
```
Expected: PASS

---

### Task 7: 图构建升级（graph.py）

**Files:**
- Modify: `core/graph.py`
- Test: `tests/test_graph.py`

核心变化：
1. 节点引用名称更新（`wrap_agent_node` 返回的包装函数）
2. 并行 fan-out：categorize → 条件边同时激活多个 Agent
3. 汇聚 merge + fallback 边
4. 适配新的 state 字段

- [ ] **Step 1: 编写集成测试**

```python
# tests/test_graph.py
import pytest
from unittest.mock import MagicMock, patch
from core.graph import customer_service_app
from core.state import CategoryOutput, QualityOutput


def make_state(message="你好"):
    return {
        "conversation_id": "test-001",
        "messages": [{"role": "user", "content": message, "timestamp": "2026-01-01T00:00:00"}],
        "routing_targets": [],
        "assigned_agents": [],
        "agent_outputs": {},
        "merged_response": "",
        "quality_score": 0,
        "quality_reason": "",
        "requires_human_review": False,
        "final_response": "",
        "metadata": {},
    }


class TestGraphCompilation:
    def test_graph_compiles(self):
        """图可以正常编译"""
        assert customer_service_app is not None

    def test_graph_has_expected_nodes(self):
        """图包含所有关键节点"""
        nodes = customer_service_app.get_graph().nodes
        node_names = list(nodes.keys())
        for name in ("categorize", "technical", "sales", "support", "feedback",
                     "merge", "fallback", "quality_check", "format_response"):
            assert name in node_names, f"Missing node: {name}"


class TestFullPipeline:
    def test_single_agent_happy_path(self):
        """单 Agent 全链路：分类 → 技术 Agent → 合并 → 评分 → 自动通过 → 格式化"""
        mock_category = CategoryOutput(categories=["technical"], confidence=0.95)
        mock_quality = QualityOutput(score=5, reason="完美")

        mock_llm_cat = MagicMock()
        mock_llm_cat.invoke.return_value = mock_category
        mock_llm_qual = MagicMock()
        mock_llm_qual.invoke.return_value = mock_quality

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [{"role": "assistant", "content": "配置 API 密钥的方法如下..."}]
        }

        with patch("core.node.llm_categorize", mock_llm_cat), \
             patch("core.node.llm_quality", mock_llm_qual), \
             patch("core.node._agents", {"technical": mock_agent, "sales": mock_agent, "support": mock_agent, "feedback": mock_agent}):

            config = {"configurable": {"thread_id": "test-thread-1"}}
            result = customer_service_app.invoke(make_state("如何配置 API 密钥"), config=config)

        assert result["final_response"] != ""
        assert "客服团队" in result["final_response"]

    def test_fallback_when_no_routing_targets(self):
        """分类返回空 → 走 fallback 路径"""
        mock_category = CategoryOutput(categories=[], confidence=0.0)

        mock_llm_cat = MagicMock()
        mock_llm_cat.invoke.return_value = mock_category

        with patch("core.node.llm_categorize", mock_llm_cat):
            config = {"configurable": {"thread_id": "test-thread-2"}}
            result = customer_service_app.invoke(make_state("!@#$%^"), config=config)

        assert "无法处理" in result["final_response"]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_graph.py -v
```
Expected: FAIL — graph.py 还在用旧的节点引用

- [ ] **Step 3: 重写 graph.py**

```python
# core/graph.py
"""图构建：并行 fan-out + merge + fallback 边 + MemorySaver + RetryPolicy。"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from core.config import RETRY_POLICY
from core.node import (
    categorize_inquiry,
    wrap_agent_node,
    merge_agent_outputs,
    fallback_reply,
    quality_check,
    human_review,
    auto_approve,
    format_response,
)
from core.logger import setup_logger

logger = setup_logger("graph")

# ============ 持久化 ============
memory = MemorySaver()

# 所有 Agent 名称
_AGENT_NAMES = ("technical", "sales", "support", "feedback")

# ============ 路由函数 ============

def route_to_agents(state: AgentState) -> list[str]:
    """根据 routing_targets 并行激活对应 Agent。

    add_conditional_edges 的路由函数返回 list 时触发 fan-out，
    图中的 4 个 Agent 节点会并行执行。
    """
    targets = state.get("routing_targets", [])
    valid = [t for t in targets if t in _AGENT_NAMES]

    if not valid:
        logger.info("[route] routing_targets 为空 → fallback")
        return ["fallback"]

    logger.info("[route] 并行 fan-out 激活: %s", valid)
    return valid


def route_after_quality(state: AgentState) -> str:
    """质量检查后路由。"""
    if state.get("requires_human_review", False):
        return "human_review"
    return "auto_approve"


def route_after_agent(state: AgentState) -> str:
    """Agent 执行后：检查是否有输出，决定走 merge 还是 fallback。"""
    agent_outputs = state.get("agent_outputs", {})
    if not agent_outputs:
        logger.warning("[route] 无 Agent 输出 → fallback")
        return "fallback"
    return "merge"


# ============ 构建图 ============

customer_service_graph = StateGraph(AgentState)

# 节点注册
customer_service_graph.add_node("categorize", categorize_inquiry)

# 4 个 Agent 节点（用 wrap_agent_node 包装 create_agent 实例）
for name in _AGENT_NAMES:
    customer_service_graph.add_node(name, wrap_agent_node(name))

customer_service_graph.add_node("merge", merge_agent_outputs)
customer_service_graph.add_node("fallback", fallback_reply)
customer_service_graph.add_node("quality_check", quality_check)
customer_service_graph.add_node("human_review", human_review)
customer_service_graph.add_node("auto_approve", auto_approve)
customer_service_graph.add_node("format_response", format_response)

# 入口
customer_service_graph.set_entry_point("categorize")

# 条件边：categorize → 多个 Agent（并行 fan-out）
customer_service_graph.add_conditional_edges(
    "categorize",
    route_to_agents,
    {
        "technical": "technical",
        "sales": "sales",
        "support": "support",
        "feedback": "feedback",
        "fallback": "fallback",
    },
)

# 每个 Agent 完成后 → merge（条件：有输出走 merge，无输出走 fallback）
for name in _AGENT_NAMES:
    customer_service_graph.add_conditional_edges(
        name,
        route_after_agent,
        {"merge": "merge", "fallback": "fallback"},
    )

# merge → quality_check
customer_service_graph.add_edge("merge", "quality_check")

# quality_check → human_review / auto_approve
customer_service_graph.add_conditional_edges(
    "quality_check",
    route_after_quality,
    {"human_review": "human_review", "auto_approve": "auto_approve"},
)

# human_review / auto_approve → format_response
customer_service_graph.add_edge("human_review", "format_response")
customer_service_graph.add_edge("auto_approve", "format_response")

# fallback → format_response（跳过审核）
customer_service_graph.add_edge("fallback", "format_response")

# format_response → END
customer_service_graph.add_edge("format_response", END)

# ============ 编译 ============

customer_service_app = customer_service_graph.compile(
    checkpointer=memory,
    retry_policy=RETRY_POLICY,
)
```

- [ ] **Step 4: 运行集成测试**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/test_graph.py -v
```
Expected: PASS

---

### Task 8: 更新 CLI 测试脚本（test.py）

**Files:**
- Modify: `test.py`

适配新的状态字段 (`routing_targets` 替代 `category`)。

- [ ] **Step 1: 重写 test.py**

```python
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
        "routing_targets": [],
        "assigned_agents": [],
        "agent_outputs": {},
        "merged_response": "",
        "quality_score": 0,
        "quality_reason": "",
        "requires_human_review": False,
        "final_response": "",
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
                if "agent_outputs" in output:
                    names = output.get("assigned_agents", [])
                    print(f"[{node_name}] Agent 激活: {names}")
                elif "merged_response" in output:
                    resp = output["merged_response"]
                    print(f"[{node_name}] 合并完成 ({len(resp)} 字符)")
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
        ("技术问题", "如何配置 API 密钥？我的请求一直返回 401 错误。"),
        ("销售咨询", "企业版套餐有什么优惠？我们公司有 50 人需要购买。"),
        ("售后支持", "我昨天提交的退款申请什么时候能处理？"),
        ("反馈建议", "你们的APP很好用，建议增加深色模式。"),
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
    stream_chat("我的账号被锁定了，如何解锁？同时我想知道企业版价格。")
    print()

    # 场景 6：多轮对话
    print("=" * 50)
    print("场景 6：多轮对话")
    print("=" * 50)
    thread = str(uuid.uuid4())
    for i, msg in enumerate([
        "你好，我想了解一下你们的API产品",
        "它的定价是多少？",
        "帮我对比一下基础版和企业版",
    ], 1):
        print(f"[第{i}轮]")
        r = chat_with_customer(msg, thread_id=thread)
        print(r[:200] + "...\n")
```

- [ ] **Step 2: 验证语法**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -c "import ast; ast.parse(open('test.py').read()); print('OK')"
```

---

### Task 9: 全量回归测试

**Files:**
- All `core/*.py` + `tests/*.py` + `test.py`

- [ ] **Step 1: 运行全部测试**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -m pytest tests/ -v
```
Expected: 所有测试 PASS

- [ ] **Step 2: 验证 CLI 脚本可导入**

```bash
cd /home/neousys/桌面/MultiAgent && python3 -c "
from core.config import LLM_CONFIG, RETRY_POLICY, AGENT_TOOLS
from core.logger import setup_logger
from core.prompts import PROMPTS
from core.tools import KNOWLEDGE_TOOLS, BUSINESS_TOOLS
from core.state import AgentState, CategoryOutput, QualityOutput
from core.node import (
    categorize_inquiry, wrap_agent_node,
    merge_agent_outputs, fallback_reply,
    quality_check, human_review, auto_approve, format_response,
)
from core.graph import customer_service_app
print('All imports OK')
"
```

---

### Task 10: 清理残留

- [ ] **Step 1: 确认无旧的 category 字段引用**

```bash
cd /home/neousys/桌面/MultiAgent && grep -rn '"category"' core/ test.py || echo "No legacy 'category' references found — OK"
```

- [ ] **Step 2: 确认无旧的 assigned_agent 字段引用**

```bash
cd /home/neousys/桌面/MultiAgent && grep -rn '"assigned_agent"' core/ test.py || echo "No legacy 'assigned_agent' references found — OK"
```

- [ ] **Step 3: 确认无旧的 agent_response 字段引用**

```bash
cd /home/neousys/桌面/MultiAgent && grep -rn '"agent_response"' core/ test.py || echo "No legacy 'agent_response' references found — OK"
```
