# rag/context.py
"""上下文构建器：将检索到的 Document 格式化为结构化文本。"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContextBuilder:
    """上下文构建器。

    将 Document 列表格式化为带元数据头的结构化字符串。

    用法:
        context = ContextBuilder.build_context(docs, max_length=2000)
    """

    @staticmethod
    def build_context(docs: List[Document], max_length: int = 2000) -> str:
        """将检索到的文档块格式化为结构化的上下文字符串。

        输出格式:
            【文档 1】 文件名 | 来源: path | 相关性: 0.0321
            文档内容...

            ==================================================
            【文档 2】 ...
        """
        if not docs:
            return "暂无相关知识。"

        parts = []
        length = 0
        separator = "\n" + "=" * 50 + "\n"

        for i, doc in enumerate(docs, 1):
            meta = f"【文档 {i}】"
            if "file_name" in doc.metadata:
                meta += f" {doc.metadata['file_name']}"
            if "source" in doc.metadata:
                meta += f" | 来源: {doc.metadata['source']}"
            if "rrf_score" in doc.metadata:
                meta += f" | 相关性: {doc.metadata['rrf_score']:.4f}"

            block = f"{meta}\n{doc.page_content}"

            if length + len(block) > max_length:
                break

            parts.append(block)
            length += len(block)

        return separator.join(parts)
