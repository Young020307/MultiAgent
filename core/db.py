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
