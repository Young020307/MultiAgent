# 保险多智能体客服系统 — 业务迁移设计

> 2026-05-14 | 架构不变，业务从 SaaS 客服 → 保险客服

---

## 1. 目标

将当前 SaaS 多智能体客服系统迁移为保险多智能体客服系统。架构、图流、RAG、人工交接全部不变，仅替换业务领域：

- 下游业务 Agent 从 technical / sales / support / feedback 改为保险四 Agent
- 数据库从 SaaS 模型改为保险模型
- 提示词从通用客服改为保险场景
- greeting + ambiguous 合并为 general

---

## 2. 命名映射

| 原名称 | 新名称 |
|--------|--------|
| `technical` | `insurance_consultation` |
| `sales` | `policy_service` |
| `support` | `claims_assistance` |
| `feedback` | `renewal_addon` |
| `greeting` + `ambiguous` | `general` |

### 2.1 IntentType

```python
IntentType = Literal[
    "general",                   # 闲聊/模糊咨询
    "insurance_consultation",    # 投保咨询
    "policy_service",            # 保单服务
    "claims_assistance",         # 理赔协助
    "renewal_addon",             # 续保加保
    "escalate",                  # 转人工（不变）
]
```

### 2.2 图节点

```python
_BUSINESS_AGENTS = (
    "insurance_consultation",
    "policy_service",
    "claims_assistance",
    "renewal_addon",
)
```

---

## 3. 数据库模型

删除原 SaaS 五表，新建保险四表。`customer_feedback` 保留不变。

```python
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
```

---

## 4. 工具分配

### 4.1 知识检索（不变）

```python
KNOWLEDGE_TOOLS = [search_knowledge_base]
```

### 4.2 业务工具

| Agent | 工具函数 | 说明 |
|-------|---------|------|
| `insurance_consultation` | `search_knowledge_base` | 查保险产品知识库 |
| `policy_service` | `db_lookup_policy`, `search_knowledge_base` | 查保单 + RAG |
| `claims_assistance` | `db_lookup_claim`, `db_submit_claim`, `search_knowledge_base` | 查/提交理赔 + RAG |
| `renewal_addon` | `db_check_renewal`, `db_query_product`, `search_knowledge_base` | 查续保/产品 + RAG |

### 4.3 工具函数签名

```python
def db_lookup_policy(policy_no: str) -> str:
    """按保单号查询保单状态、有效期、持有人等"""

def db_lookup_claim(claim_no: str) -> str:
    """按理赔号查询理赔进度和详情"""

def db_submit_claim(policy_no: str, reason: str, amount: float) -> str:
    """为指定保单提交一份理赔申请，返回理赔号"""

def db_check_renewal(policy_no: str) -> str:
    """查询某保单的续保方案和可用优惠"""

def db_query_product(product_name: str = "") -> str:
    """查询保险产品详情；product_name 为空时列出全部产品"""

def db_log_feedback(content: str, contact: str = "") -> str:
    """记录用户反馈（不变）"""
```

### 4.4 工具注册表

```python
BUSINESS_TOOLS = {
    "insurance_consultation": [],
    "policy_service":     [db_lookup_policy],
    "claims_assistance":  [db_lookup_claim, db_submit_claim],
    "renewal_addon":      [db_check_renewal, db_query_product],
}

AGENT_TOOLS = {
    "insurance_consultation": ["search_knowledge_base"],
    "policy_service":         ["db_lookup_policy", "search_knowledge_base"],
    "claims_assistance":      ["db_lookup_claim", "db_submit_claim", "search_knowledge_base"],
    "renewal_addon":          ["db_check_renewal", "db_query_product", "search_knowledge_base"],
}
```

---

## 5. 提示词变更

### 5.1 Router

意图类型改为：
- `general`: 打招呼、闲聊、需求模糊
- `insurance_consultation`: 咨询保险产品、投保条件、保费
- `policy_service`: 查保单、变更保单、保单相关问题
- `claims_assistance`: 出险报案、理赔进度、赔付问题
- `renewal_addon`: 续保、加保、升级保障
- `escalate`: 明确要求转人工（不变）

### 5.2 业务 Agent

- **insurance_consultation**: 保险咨询顾问，解答投保条件、保费、保障范围，对比产品，不出单
- **policy_service**: 保单服务专员，查询保单状态、变更信息、提醒续期
- **claims_assistance**: 理赔协助专员，指导报案流程、查询理赔进度、说明赔付标准
- **renewal_addon**: 续保加保顾问，推荐续保方案、介绍加保产品、介绍优惠

### 5.3 不变

`conversation_agent`、`quality` 提示词逻辑不变，只改措辞。

---

## 6. 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `core/state.py` | 修改 | `IntentType` 改 6 值 |
| `core/config.py` | 修改 | Agent key、工具映射 |
| `core/prompts.py` | 修改 | 全部 prompt 改为保险场景 |
| `core/db.py` | 重写 | 新建保险四表，`init_db` 不变 |
| `core/tools.py` | 重写 | 新工具函数 + 注册表 |
| `core/graph.py` | 修改 | `_BUSINESS_AGENTS` 元组 |
| `core/node.py` | 修改 | 路由目标引用 |
| `core/agent_factory.py` | 修改 | Agent 创建 key |
| `test/test.py` | 修改 | 场景测试更新 |
| `test/chat.py` | 修改 | 节点名称引用 |
| `test/test_rag.py` | 不变 | |
| `test/test_handoff_registry.py` | 不变 | |
| `rag/` | 不变 | 全部不变 |
| `knowledge/` | 替换 | 保险知识文档（用户提供） |
| `streamlit_app.py` | 不变 | |
| `pages/agent_dashboard.py` | 不变 | |

---

## 7. 依赖和环境

- 已有依赖不变
- `DATABASE_URL` 可改为 `sqlite:///insurance_service.db`（不改也能用）

---

## 8. 不在范围

- 不出单、不支付、不做真实承保
- 知识库文档内容由用户自行准备
- 多语言、合规审核
