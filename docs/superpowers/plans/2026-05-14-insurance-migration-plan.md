# 保险多智能体客服系统 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 SaaS 客服系统迁移为保险客服系统，架构不变，替换业务领域（4 个业务 Agent、数据库模型、提示词、工具）

**Architecture:** 架构零改动。LangGraph 图流、RAG 管线、人工交接、情感检测、异常计数器全部保持。仅替换 IntentType 枚举值、Agent 名称、DB 模型、工具函数、提示词

**Tech Stack:** Python 3.12, LangGraph 1.x, SQLAlchemy, Streamlit 1.57

---

## 文件变更总览

| 文件 | 改动类型 |
|------|---------|
| `core/state.py` | 修改 IntentType |
| `core/config.py` | 修改 Agent 名称、工具映射、DB URL |
| `core/db.py` | 重写 ORM 模型 |
| `core/tools.py` | 重写业务工具函数 |
| `core/prompts.py` | 重写全部提示词 |
| `core/graph.py` | 修改 `_BUSINESS_AGENTS` |
| `core/node.py` | 修改 business 列表 |
| `core/agent_factory.py` | 修改 Agent 名称引用 |
| `test/test.py` | 修改测试场景 |
| `test/chat.py` | 修改节点名引用 |

---

### Task 1: `core/state.py` + `core/config.py` — 基础命名 & 配置

**职责:** 修改 IntentType 枚举 + Agent LLM 配置 key + 工具映射表，是整个迁移的类型基础

**依赖:** 无

---

- [ ] **Step 1: 修改 `core/state.py` — IntentType**

```python
# core/state.py — 修改 IntentType 行

# 原来:
IntentType = Literal["greeting", "ambiguous", "technical", "sales", "support", "feedback", "escalate"]

# 改为:
IntentType = Literal["general", "insurance_consultation", "policy_service", "claims_assistance", "renewal_addon", "escalate"]
```

同时修改 `RouterDecision.intent` 的 `description`:

```python
# 原来:
intent: IntentType = Field(
    default="ambiguous",
    description="greeting|ambiguous|technical|sales|support|feedback",
)

# 改为:
intent: IntentType = Field(
    default="general",
    description="general|insurance_consultation|policy_service|claims_assistance|renewal_addon",
)
```

- [ ] **Step 2: 修改 `core/config.py` — AGENT_LLM_CONFIG**

替换所有业务 Agent key:

```python
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
```

- [ ] **Step 3: 修改 `core/config.py` — AGENT_TOOLS**

```python
# 原来:
AGENT_TOOLS = {
    "conversation_agent": [],
    "technical": ["search_knowledge_base"],
    "sales":     ["db_query_pricing", "db_check_promotion", "search_knowledge_base"],
    "support":   ["db_lookup_order", "db_check_refund_status", "search_knowledge_base"],
    "feedback":  ["db_log_feedback"],
}

# 改为:
AGENT_TOOLS = {
    "conversation_agent": [],
    "insurance_consultation": ["search_knowledge_base"],
    "policy_service":         ["db_lookup_policy", "search_knowledge_base"],
    "claims_assistance":      ["db_lookup_claim", "db_submit_claim", "search_knowledge_base"],
    "renewal_addon":          ["db_check_renewal", "db_query_product", "search_knowledge_base"],
}
```

- [ ] **Step 4: 修改 `core/config.py` — DATABASE_URL**

```python
# 原来:
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///customer_service.db")

# 改为:
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///insurance_service.db")
```

- [ ] **Step 5: 验证导入**

```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. python3 -c "
from core.state import AgentState, IntentType, RouterDecision
print('IntentType:', IntentType.__args__)
from core.config import AGENT_LLM_CONFIG, AGENT_TOOLS, DATABASE_URL
print('Agent keys:', list(AGENT_LLM_CONFIG.keys()))
print('Tool keys:', list(AGENT_TOOLS.keys()))
print('DB URL:', DATABASE_URL)
"
```

Expected: 输出新的 6 个 IntentType 值、Agent key 列表含 `insurance_consultation` 等、DB URL 含 `insurance_service`

---

### Task 2: `core/db.py` — 保险数据库模型

**职责:** 将 SaaS 五表替换为保险四表，保留 `init_db()`、`get_db()`、`CustomerFeedback`

**依赖:** 无

---

- [ ] **Step 1: 重写 `core/db.py`**

完整替换 ORM 模型部分（`PricingPlan`, `Promotion`, `Order`, `Refund` → `InsuranceProduct`, `Policy`, `Claim`, `RenewalPlan`），`CustomerFeedback` 保留不变，`init_db()` 和 `get_db()` 保持不变。

```python
# core/db.py
"""SQLAlchemy 数据库层：保险 ORM 模型 + 会话管理。

使用前需设置 DATABASE_URL 环境变量，默认为本地 SQLite。"""

import os

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///insurance_service.db")

_engine = create_engine(DATABASE_URL, echo=False)
_SessionLocal = sessionmaker(bind=_engine)
Base = declarative_base()

# ================================================================
# ORM 模型
# ================================================================

class InsuranceProduct(Base):
    __tablename__ = "insurance_products"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    category = Column(String(64), nullable=False)
    premium_yearly = Column(Float, nullable=False)
    coverage_detail = Column(Text, default="")
    min_age = Column(Integer, default=0)
    max_age = Column(Integer, default=100)


class Policy(Base):
    __tablename__ = "policies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_no = Column(String(32), unique=True, nullable=False, index=True)
    holder_name = Column(String(64), nullable=False)
    product_name = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)


class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_no = Column(String(32), unique=True, nullable=False, index=True)
    policy_no = Column(String(32), nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String(32), nullable=False)
    filed_date = Column(DateTime, nullable=False)
    notes = Column(Text, default="")


class RenewalPlan(Base):
    __tablename__ = "renewal_plans"
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32), unique=True, nullable=False, index=True)
    product_name = Column(String(128), nullable=False)
    discount_desc = Column(Text, nullable=False)
    is_active = Column(Integer, default=1)


class CustomerFeedback(Base):
    __tablename__ = "customer_feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    contact = Column(String(128), default="")
    created_at = Column(DateTime)


# ================================================================
# 会话管理
# ================================================================

def init_db():
    """初始化数据库表（首次运行时调用）。"""
    Base.metadata.create_all(bind=_engine)


def get_db() -> Session:
    """获取一个数据库会话。"""
    return _SessionLocal()
```

- [ ] **Step 2: 验证模型可以创建表**

```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. python3 -c "
from core.db import init_db, Base
init_db()
print('Tables:', list(Base.metadata.tables.keys()))
"
```

Expected: `Tables: ['insurance_products', 'policies', 'claims', 'renewal_plans', 'customer_feedback']`

- [ ] **Step 3: 删除旧的数据库文件**

```bash
rm -f /home/neousys/桌面/MultiAgent/customer_service.db
```

---

### Task 3: `core/tools.py` — 保险业务工具

**职责:** 将 SaaS 工具替换为 6 个保险工具函数 + 更新注册表。RAG 相关函数（`search_knowledge_base`, `configure_rag`, `get_low_conf_streak`, `reset_low_conf_streak`）保留不变

**依赖:** Task 2（需要数据库模型）

---

- [ ] **Step 1: 重写 `core/tools.py` 中的业务工具部分**

完整替换业务工具函数和注册表。保留 RAG 相关所有函数。

```python
# core/tools.py
"""知识检索（RAG） + 保险业务数据库操作工具。"""

import json
from datetime import datetime

from core.db import get_db, init_db
from core.db import (
    InsuranceProduct,
    Policy,
    Claim,
    RenewalPlan,
    CustomerFeedback,
)
from core.logger import setup_logger
from core.config import MAX_RETRIEVAL_ROUNDS

logger = setup_logger("tools")

# ================================================================
# 知识检索 — Agentic RAG（不变）
# ================================================================

_rag_client = None
_retrieval_counts: dict[str, int] = {}
_consecutive_low_conf: dict[str, int] = {}


def configure_rag(client):
    """配置 RAG 客户端。（不变）"""
    global _rag_client
    _rag_client = client
    logger.info("[tools] RAG 客户端已配置")


def search_knowledge_base(query: str, conversation_id: str = "") -> str:
    """Agentic RAG 检索工具。（不变）"""
    if _rag_client is None:
        return json.dumps({
            "results": [],
            "max_score": 0,
            "low_confidence": True,
            "error": "知识库尚未配置，无法检索。",
        }, ensure_ascii=False)

    if conversation_id:
        if _consecutive_low_conf.get(conversation_id, 0) >= 2:
            return json.dumps({
                "results": [],
                "max_score": 0,
                "low_confidence": True,
                "error": "前两次检索均为低置信度，已停止继续检索。",
            }, ensure_ascii=False)
        if _retrieval_counts.get(conversation_id, 0) >= MAX_RETRIEVAL_ROUNDS:
            return json.dumps({
                "results": [],
                "max_score": 0,
                "low_confidence": True,
                "error": f"检索次数已达上限({MAX_RETRIEVAL_ROUNDS}次)。",
            }, ensure_ascii=False)

    try:
        result_json = _rag_client.search(query)

        if conversation_id:
            _retrieval_counts[conversation_id] = _retrieval_counts.get(conversation_id, 0) + 1
            data = json.loads(result_json)
            if data.get("low_confidence"):
                _consecutive_low_conf[conversation_id] = _consecutive_low_conf.get(conversation_id, 0) + 1
            else:
                _consecutive_low_conf[conversation_id] = 0

        return result_json
    except Exception as e:
        logger.error("[tools] RAG 检索失败: %s", e)
        return json.dumps({
            "results": [],
            "max_score": 0,
            "low_confidence": True,
            "error": f"知识库检索异常：{e}",
        }, ensure_ascii=False)


def get_low_conf_streak(conversation_id: str) -> int:
    """（不变）"""
    return _consecutive_low_conf.get(conversation_id, 0)


def reset_low_conf_streak(conversation_id: str):
    """（不变）"""
    if conversation_id in _consecutive_low_conf:
        _consecutive_low_conf[conversation_id] = 0


# ================================================================
# 保险业务数据库操作
# ================================================================

def db_query_product(product_name: str = "") -> str:
    """查询保险产品详情。

    不传参数时列出全部产品，传名称时模糊匹配。
    """
    db = get_db()
    try:
        if not product_name.strip():
            rows = db.query(InsuranceProduct).all()
            if not rows:
                return "暂无保险产品信息。"
            lines = []
            for r in rows:
                lines.append(
                    f"【{r.name}】{r.category} — ¥{r.premium_yearly}/年，"
                    f"投保年龄 {r.min_age}-{r.max_age} 岁。{r.coverage_detail}"
                )
            return "\n".join(lines)

        name_lower = product_name.lower().strip()
        rows = db.query(InsuranceProduct).filter(
            InsuranceProduct.name.ilike(f"%{name_lower}%")
        ).all()
        if not rows:
            all_products = db.query(InsuranceProduct.name).all()
            names = ", ".join(p.name for p in all_products) if all_products else "暂无产品"
            return f"未找到「{product_name}」产品。可选产品：{names}"

        r = rows[0]
        return (
            f"【{r.name}】{r.category} — ¥{r.premium_yearly}/年，"
            f"投保年龄 {r.min_age}-{r.max_age} 岁。\n{r.coverage_detail}"
        )
    finally:
        db.close()


def db_lookup_policy(policy_no: str) -> str:
    """按保单号查询保单状态。"""
    db = get_db()
    try:
        policy_no_upper = policy_no.upper().strip()
        row = db.query(Policy).filter(Policy.policy_no == policy_no_upper).first()

        if row is None:
            return f"未找到保单「{policy_no}」。请确认保单号是否正确。"
        return (
            f"保单 {row.policy_no}：产品「{row.product_name}」，"
            f"持有人 {row.holder_name}，状态「{row.status}」，"
            f"有效期 {row.start_date.strftime('%Y-%m-%d')} 至 {row.end_date.strftime('%Y-%m-%d')}。"
        )
    finally:
        db.close()


def db_lookup_claim(claim_no: str) -> str:
    """按理赔号查询理赔进度。"""
    db = get_db()
    try:
        claim_no_upper = claim_no.upper().strip()
        row = db.query(Claim).filter(Claim.claim_no == claim_no_upper).first()

        if row is None:
            return f"未找到理赔单「{claim_no}」。请确认理赔号是否正确。"

        filed = row.filed_date.strftime("%Y-%m-%d") if row.filed_date else "未知"
        result = (
            f"理赔号 {row.claim_no}：保单 {row.policy_no}，"
            f"金额 ¥{row.amount}，状态「{row.status}」，提交于 {filed}。"
        )
        if row.notes:
            result += f"\n备注：{row.notes}"
        return result
    finally:
        db.close()


def db_submit_claim(policy_no: str, reason: str, amount: float) -> str:
    """提交一份理赔申请。"""
    db = get_db()
    try:
        policy_no_upper = policy_no.upper().strip()
        policy = db.query(Policy).filter(Policy.policy_no == policy_no_upper).first()
        if policy is None:
            return f"未找到保单「{policy_no}」，无法提交理赔。请确认保单号。"

        import uuid
        claim_no = "CLM" + str(uuid.uuid4())[:8].upper()
        claim = Claim(
            claim_no=claim_no,
            policy_no=policy_no_upper,
            amount=amount,
            status="已受理",
            filed_date=datetime.now(),
            notes=reason,
        )
        db.add(claim)
        db.commit()
        return (
            f"理赔申请已提交。理赔号 {claim_no}，"
            f"保单 {policy_no_upper}，金额 ¥{amount}，当前状态「已受理」。"
            f"预计 5-10 个工作日完成审核。"
        )
    except Exception as e:
        db.rollback()
        logger.error("[tools] 提交理赔失败: %s", e)
        return "理赔申请提交失败，请稍后重试或联系人工客服。"
    finally:
        db.close()


def db_check_renewal(policy_no: str) -> str:
    """查询某保单关联的续保方案和优惠。"""
    db = get_db()
    try:
        policy_no_upper = policy_no.upper().strip()
        policy = db.query(Policy).filter(Policy.policy_no == policy_no_upper).first()
        if policy is None:
            return f"未找到保单「{policy_no}」。请确认保单号。"

        renewals = db.query(RenewalPlan).filter(
            RenewalPlan.product_name == policy.product_name,
            RenewalPlan.is_active == 1,
        ).all()

        result = f"保单 {policy_no_upper}（{policy.product_name}）的续保信息：\n"
        if not renewals:
            renewals = db.query(RenewalPlan).filter(RenewalPlan.is_active == 1).all()
            if renewals:
                result += "该产品暂无专属续保方案，以下是通用续保方案：\n"
            else:
                return result + "暂无可用续保方案，请联系客服。"
        for r in renewals:
            result += f"  - [{r.code}] {r.discount_desc}\n"
        return result.strip()
    finally:
        db.close()


def db_log_feedback(content: str, contact: str = "") -> str:
    """记录用户反馈到数据库。（不变）"""
    db = get_db()
    try:
        feedback = CustomerFeedback(
            content=content,
            contact=contact,
            created_at=datetime.now(),
        )
        db.add(feedback)
        db.commit()
        return f"已记录用户反馈「{content}」。感谢您的建议，产品团队会认真评估。"
    except Exception as e:
        db.rollback()
        logger.error("[tools] 记录反馈失败: %s", e)
        return "反馈记录失败，请稍后重试。"
    finally:
        db.close()


# ================================================================
# 工具注册表
# ================================================================

KNOWLEDGE_TOOLS = [search_knowledge_base]

BUSINESS_TOOLS = {
    "insurance_consultation": [],
    "policy_service":     [db_lookup_policy],
    "claims_assistance":  [db_lookup_claim, db_submit_claim],
    "renewal_addon":      [db_check_renewal, db_query_product],
}
```

- [ ] **Step 2: 验证导入**

```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. python3 -c "
from core.tools import (
    search_knowledge_base, configure_rag, get_low_conf_streak, reset_low_conf_streak,
    db_query_product, db_lookup_policy, db_lookup_claim, db_submit_claim, db_check_renewal,
    db_log_feedback, BUSINESS_TOOLS, KNOWLEDGE_TOOLS,
)
print('Import OK')
print('BUSINESS_TOOLS:', {k: [f.__name__ for f in v] for k, v in BUSINESS_TOOLS.items()})
"
```

Expected: `Import OK` + 各 Agent 工具列表

---

### Task 4: `core/prompts.py` — 保险场景提示词

**职责:** 将所有 System Prompt 改写为保险领域。Router 意图说明改保险场景，4 个业务 Agent 改为保险角色

**依赖:** Task 1（需要新 Agent 名称）

---

- [ ] **Step 1: 重写 `core/prompts.py`**

```python
# core/prompts.py
"""集中管理所有提示词模板。模板使用 .format(**ctx) 填入上下文。"""

PROMPTS = {
    "router": """判断客户消息的意图类型。

{context}

当前客户消息：{user_message}

类型说明：
- general: 打招呼、闲聊、需求模糊无法确定具体业务
- insurance_consultation: 咨询保险产品、投保条件、保费、保障范围
- policy_service: 查保单、变更保单信息、保单相关问题
- claims_assistance: 出险报案、理赔进度查询、赔付标准
- renewal_addon: 续保、加保、升级保障、优惠咨询
- escalate: 用户明确要求转人工，如"转人工"、"我要跟真人说话"、"有没有人工客服"

严格返回 JSON（不要 markdown 包裹）：
{{"intent": "general", "confidence": 0.9}}""",

    "conversation_agent": """你是保险客服主管，负责接待客户的问候、闲聊和模糊咨询。

你的职责：
- 问候/闲聊：友好回应，主动介绍你能提供的帮助（查保单、问理赔、咨询产品等）
- 模糊意图：生成选择题帮客户快速消歧，例如 "请问您是需要 A.咨询保险产品 B.查询保单 C.理赔协助 还是 D.续保加保？"
- 如果 clarify_count >= 2 仍无法明确意图：给出安全兜底话术，建议联系人工客服

客户消息中会附带 intent 和 clarify_count 信息，请据此做出恰当回应。""",

    "insurance_consultation": """你是一位专业的保险咨询顾问，专门解答保险产品相关问题。
你的回答应该：
- 准确、详细、专业
- 清晰说明投保条件、保费、保障范围
- 对比不同产品的优劣，帮客户做出选择
- 主动提醒注意事项（免责条款、等待期等）

{context}

用户问题：{user_message}
请提供专业、详细的保险咨询回复。

你拥有 search_knowledge_base 工具。调用后你会得到一个 JSON：
- 如果 low_confidence 为 true，不要编造，告知用户"没有找到足够相关的信息，建议联系人工客服"。
- 如果 low_confidence 为 false，使用 results 中的 text 生成回答，优先引用分数(score)高的结果。""",

    "policy_service": """你是一位耐心的保单服务专员，处理客户的保单查询和变更需求。
你的回答应该：
- 表达理解和专业态度
- 清晰说明保单状态和信息
- 提供具体的操作指引
- 提醒保单到期日等重要信息

{context}

用户问题：{user_message}
请提供周到、专业的保单服务回复。

你拥有 search_knowledge_base 工具。调用后你会得到一个 JSON：
- 如果 low_confidence 为 true，不要编造，告知用户"没有找到足够相关的信息，建议联系人工客服"。
- 如果 low_confidence 为 false，使用 results 中的 text 生成回答，优先引用分数(score)高的结果。""",

    "claims_assistance": """你是一位有同理心的理赔协助专员，帮助客户处理理赔相关事务。
你的回答应该：
- 首先表达关心和理解
- 清晰说明理赔流程和所需材料
- 提供理赔进度查询
- 说明赔付标准和预计时间
- 必要时引导客户拨打报案电话

{context}

用户问题：{user_message}
请提供温暖、专业的理赔协助回复。

你拥有 search_knowledge_base 工具。调用后你会得到一个 JSON：
- 如果 low_confidence 为 true，不要编造，告知用户"没有找到足够相关的信息，建议联系人工客服"。
- 如果 low_confidence 为 false，使用 results 中的 text 生成回答，优先引用分数(score)高的结果。""",

    "renewal_addon": """你是一位专业的续保加保顾问，帮助客户续保和升级保障。
你的回答应该：
- 友好、热情、有说服力
- 清晰说明续保方案和优惠
- 主动推荐加保产品，说明加保的好处
- 提醒续保时间窗口和过期影响

{context}

用户问题：{user_message}
请提供有帮助的续保加保建议。

你拥有 search_knowledge_base 工具。调用后你会得到一个 JSON：
- 如果 low_confidence 为 true，不要编造，告知用户"没有找到足够相关的信息，建议联系人工客服"。
- 如果 low_confidence 为 false，使用 results 中的 text 生成回答，优先引用分数(score)高的结果。""",

    "quality": """评估以下客服回复的质量。

咨询类别：{categories}
客服回复：{response}

评分标准（1-5）：
5-完美 4-良好 3-一般 2-较差 1-不合格

严格按以下 JSON 格式返回（不要用 markdown 代码块包裹）：
{{"score": 4, "reason": "回复准确但可以更详细"}}""",
}
```

- [ ] **Step 2: 验证导入**

```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. python3 -c "
from core.prompts import PROMPTS
print('Prompt keys:', list(PROMPTS.keys()))
print('Router prompt length:', len(PROMPTS['router']))
"
```

Expected: Prompt keys 含 6 个 key（router, conversation_agent, insurance_consultation, policy_service, claims_assistance, renewal_addon, quality）

---

### Task 5: `core/graph.py` + `core/node.py` + `core/agent_factory.py` — 图 & 节点 & Agent 工厂

**职责:** 更新所有引用旧 Agent 名称的地方

**依赖:** Task 1（需要新名称定义）

---

- [ ] **Step 1: 修改 `core/graph.py` — `_BUSINESS_AGENTS`**

```python
# core/graph.py — 修改 _BUSINESS_AGENTS 元组

# 原来:
_BUSINESS_AGENTS = ("technical", "sales", "support", "feedback")

# 改为:
_BUSINESS_AGENTS = ("insurance_consultation", "policy_service", "claims_assistance", "renewal_addon")
```

- [ ] **Step 2: 修改 `core/node.py` — supervisor_router**

修改 `supervisor_router` 函数中的 `business` 列表:

```python
# core/node.py — supervisor_router 函数中

# 原来:
business = ["technical", "sales", "support", "feedback"]

# 改为:
business = ["insurance_consultation", "policy_service", "claims_assistance", "renewal_addon"]
```

- [ ] **Step 3: 修改 `core/agent_factory.py` — 节点包装**

`wrap_agent_node` 和 `get_llm_for` 中的 Agent 名称映射需要更新。检查函数 `wrap_agent_node` 中 Agent 名称逻辑：该函数以 agent_name 为参数，动态创建 create_agent，本身不需要修改。但需要确认 `get_llm_for` 函数中的映射逻辑是否依赖硬编码名称。

查看 `get_llm_for` 的实现逻辑：它从 `AGENT_LLM_CONFIG` 按 name 查询配置。如果 key 已经是新名称（Task 1 已更新），则无需额外修改。

但需要确认 agent_factory 中是否有其他地方硬编码了旧名称。检查是否有 import 旧 Agent 名称的地方。

```bash
cd /home/neousys/桌面/MultiAgent && grep -n "technical\|sales\|support\|feedback" core/agent_factory.py || echo "No matches — OK"
```

如果无匹配，则 agent_factory.py 无需修改。如果有，需要替换。

- [ ] **Step 4: 验证导入 + 图编译**

```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. python3 -c "
from core.graph import customer_service_app
print('Graph compiled OK')
print('Nodes:', list(customer_service_app.nodes.keys()))
"
```

Expected: 图编译成功，节点列表中包含 `insurance_consultation`, `policy_service`, `claims_assistance`, `renewal_addon`

---

### Task 6: 测试文件更新

**职责:** 将 test.py 和 chat.py 中的场景和引用更新为保险领域

**依赖:** Task 1-5

---

- [ ] **Step 1: 修改 `test/test.py` — 测试场景**

替换 4 个测试场景为保险场景：

```python
# test/test.py — 修改 tests 列表（约第 72-78 行）

# 原来:
tests = [
    ("技术问题", "如何配置 API 密钥？我的请求一直返回 401 错误。"),
    ("销售咨询", "企业版套餐有什么优惠？我们公司有 50 人需要购买。"),
    ("售后支持", "我昨天提交的退款申请什么时候能处理？"),
    ("反馈建议", "你们的APP很好用，建议增加深色模式。"),
]

# 改为:
tests = [
    ("投保咨询", "我想给家里人买一份重疾险，有什么推荐的吗？"),
    ("保单服务", "帮我查一下我的保单 POL-20240001，看看什么时候到期。"),
    ("理赔协助", "我的车被刮了，我已经报了案，帮我查一下理赔进度 CLM-20240001。"),
    ("续保加保", "我的保单快到期了，有什么续保优惠吗？"),
]
```

同时修改场景 5 的流式示例和场景 6 的多轮对话示例：

```python
# 场景 5：流式输出
stream_chat("我想了解百万医疗险的保障范围是什么？")
print()

# 场景 6：多轮对话
thread = str(uuid.uuid4())
for i, msg in enumerate([
    "你好，我想咨询一下意外险",
    "保费大概多少钱？保障额度是多少？",
    "帮我对比一下基础版和升级版意外险",
], 1):
    print(f"[第{i}轮]")
    r = chat_with_customer(msg, thread_id=thread)
    print(r[:200] + "...\n")
```

- [ ] **Step 2: 修改 `test/chat.py` — 节点名引用**

chat.py 的 stream 事件打印中，节点名称 `technical`, `sales`, `support`, `feedback` 需要更新：

```python
# test/chat.py — 修改 event 处理中的节点名判断（约第 67-71 行）

# 原来:
elif node_name in ("technical", "sales", "support", "feedback"):

# 改为:
elif node_name in ("insurance_consultation", "policy_service", "claims_assistance", "renewal_addon"):
```

- [ ] **Step 3: 运行集成测试**

```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. python test/test.py
```

预期：4 个场景全部返回非空 `final_response`，流式和多轮正常执行。

---

### Task 7: 最终验证

- [ ] **Step 1: 运行所有现有测试**

```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. python -m pytest test/test_handoff_registry.py test/test_rag.py -v
```

- [ ] **Step 2: 清理旧文件**

```bash
rm -f /home/neousys/桌面/MultiAgent/customer_service.db
```

- [ ] **Step 3: 端到端验证 — chat.py**

```bash
cd /home/neousys/桌面/MultiAgent && source $(conda info --base)/etc/profile.d/conda.sh && conda activate agent && PYTHONPATH=. echo "我想咨询一下重疾险" | timeout 30 python test/chat.py || true
```
