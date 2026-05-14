# scripts/setup_db.py
"""初始化数据库：建表 + 插入示例数据。

运行方式：
    cd /home/neousys/桌面/MultiAgent
    python scripts/setup_db.py

首次运行创建 customer_service.db，再次运行跳过已有数据。
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import (
    init_db,
    get_db,
    PricingPlan,
    Promotion,
    Order,
    Refund,
    CustomerFeedback,
)

# ================================================================
# 建表
# ================================================================
print("创建数据库表...")
init_db()
print("完成。")

# ================================================================
# 插入示例数据（幂等：跳过已存在的记录）
# ================================================================
db = get_db()

# ---- 价格套餐 ----
if db.query(PricingPlan).count() == 0:
    db.add_all([
        PricingPlan(name="basic", price_monthly=99, max_users=5,
                    storage_gb=10, api_rate_limit=10,
                    description="适合个人开发者和小型项目。"),
        PricingPlan(name="professional", price_monthly=499, max_users=50,
                    storage_gb=100, api_rate_limit=100,
                    description="适合中型团队，包含优先邮件支持。"),
        PricingPlan(name="enterprise", price_monthly=1999, max_users=500,
                    storage_gb=1024, api_rate_limit=1000,
                    description="适合大型企业，包含专属技术支持和 SLA 保障。"),
    ])
    print("价格套餐数据已插入。")
else:
    print("价格套餐数据已存在，跳过。")

# ---- 优惠码 ----
if db.query(Promotion).count() == 0:
    db.add_all([
        Promotion(code="WELCOME50", description="新用户首月 5 折，适用于专业版和企业版。", is_active=1),
        Promotion(code="ANNUAL20", description="年付享 8 折优惠，适用于所有套餐。", is_active=1),
        Promotion(code="REFER10", description="推荐好友双方各得 10% 折扣。", is_active=1),
        Promotion(code="EXPIRED01", description="2025 年双十一限时优惠。", is_active=0),
    ])
    print("优惠码数据已插入。")
else:
    print("优惠码数据已存在，跳过。")

# ---- 订单 ----
if db.query(Order).count() == 0:
    db.add_all([
        Order(order_id="ORD-001", status="已发货", created_at=datetime(2026, 4, 28)),
        Order(order_id="ORD-002", status="处理中", created_at=datetime(2026, 5, 1)),
        Order(order_id="ORD-003", status="已完成", created_at=datetime(2026, 4, 15)),
        Order(order_id="ORD-004", status="已取消", created_at=datetime(2026, 3, 20)),
    ])
    print("订单数据已插入。")
else:
    print("订单数据已存在，跳过。")

# ---- 退款单 ----
if db.query(Refund).count() == 0:
    db.add_all([
        Refund(refund_id="REF-001", status="处理中", amount=499, expected_days="5-7 个工作日"),
        Refund(refund_id="REF-002", status="已完成", amount=99,
               expected_days="5-7 个工作日", completed_at=datetime(2026, 4, 30)),
        Refund(refund_id="REF-003", status="待审核", amount=1999, expected_days="3-5 个工作日"),
    ])
    print("退款单数据已插入。")
else:
    print("退款单数据已存在，跳过。")

# ---- 用户反馈 ----
if db.query(CustomerFeedback).count() == 0:
    db.add_all([
        CustomerFeedback(
            content="建议增加深色模式，夜间使用更方便。",
            contact="user001@example.com",
            created_at=datetime(2026, 4, 20),
        ),
        CustomerFeedback(
            content="API 文档的搜索功能不够精准，希望能改进。",
            contact="dev_team@example.com",
            created_at=datetime(2026, 4, 25),
        ),
    ])
    print("反馈数据已插入。")
else:
    print("反馈数据已存在，跳过。")

db.commit()
db.close()

print()
print("数据库初始化完成：customer_service.db")
