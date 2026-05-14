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
