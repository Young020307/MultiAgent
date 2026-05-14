"""
=============================================================================
RAG 检索与生成参考实现
=============================================================================
功能: 混合检索 (向量 + BM25 + RRF 重排) + 查询重写 + 上下文构建

依赖:
    pip install langchain-community langchain-core rank_bm25
    如需查询重写: pip install langchain-ollama

使用方式:
    1. 你的项目已完成: 文档摄入 + 嵌入 & 向量存储
    2. 将本文件放入你的项目
    3. 传入你已有的 vectorstore 和 document chunks 即可使用

用法示例:
    from rag_retrieval_generation import RetrievalOptimizer, ContextBuilder

    # 混合检索
    retriever = RetrievalOptimizer(vectorstore, chunks)
    docs = retriever.hybrid_search("闯红灯扣几分", top_k=3)

    # 上下文构建
    builder = ContextBuilder()
    context = builder.build_context(docs, max_length=2000)

    # 查询重写 (可选, 需要 Ollama)
    builder.setup_llm(model_name="qwen2.5:7b", base_url="http://localhost:11434")
    rewritten = builder.query_rewrite("交通规则")
=============================================================================
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma  # or your vectorstore type
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ============================================================================
# 第一部分: 混合检索 (向量检索 + BM25 关键词检索 + RRF 重排)
# ============================================================================

class RetrievalOptimizer:
    """
    混合检索优化器

    功能:
        - 并行执行向量检索 + BM25 关键词检索
        - 使用 RRF (Reciprocal Rank Fusion) 算法融合排序
        - 支持元数据过滤检索

    用法:
        optimizer = RetrievalOptimizer(vectorstore, chunks)
        docs = optimizer.hybrid_search("查询文本", top_k=3)
        docs = optimizer.metadata_filtered_search("查询", {"file_type": "pdf"}, top_k=5)
    """

    def __init__(self, vectorstore, chunks: List[Document]):
        """
        Args:
            vectorstore: 你的向量存储实例 (需支持 .as_retriever())
            chunks: 文档块列表 (List[langchain_core.documents.Document])
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self._setup_retrievers()

    # ------------------------------------------------------------------
    # 内部: 初始化两个检索器
    # ------------------------------------------------------------------

    def _setup_retrievers(self):
        """构建向量检索器 + BM25 检索器"""
        # 向量检索器 (similarity search, top_k=5 用于候选池)
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        # BM25 关键词检索器 (基于词频统计, 不依赖 embedding)
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5,
        )

        logger.info("检索器初始化完成: vector(top_k=5) + BM25(top_k=5)")

    # ------------------------------------------------------------------
    # 混合检索入口
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索: 向量检索 + BM25 并行执行, RRF 融合后返回 top_k

        Args:
            query: 用户查询
            top_k:  返回的结果数量

        Returns:
            重排后的 Document 列表, metadata 中包含 'rrf_score' 字段
        """
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        reranked = self._rrf_rerank(vector_docs, bm25_docs, k=60)
        return reranked[:top_k]

    # ------------------------------------------------------------------
    # 带元数据过滤的检索
    # ------------------------------------------------------------------

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        """
        先混合检索扩大候选池 (top_k * 3), 再按元数据过滤

        Args:
            query:   用户查询
            filters: 过滤条件, 如 {"file_type": "pdf"} 或 {"doc_type": ["law", "manual"]}
            top_k:   返回的结果数量

        Returns:
            过滤后的 Document 列表
        """
        candidates = self.hybrid_search(query, top_k * 3)
        filtered = []

        for doc in candidates:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                else:
                    if doc.metadata[key] != value:
                        match = False
                        break
            if match:
                filtered.append(doc)
                if len(filtered) >= top_k:
                    break

        return filtered

    # ------------------------------------------------------------------
    # RRF 重排核心算法
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_rerank(
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion) 融合排序

        公式:
            score(d) = 1/(k + rank_vector(d)) + 1/(k + rank_bm25(d))

        Args:
            vector_docs: 向量检索结果 (排序相关)
            bm25_docs:   BM25 检索结果 (排序相关)
            k:           平滑参数, 默认 60 (经验值, 可调)

        Returns:
            按融合分数降序排列的 Document 列表 (含 rrf_score 元数据)
        """
        doc_scores: Dict[int, float] = {}
        doc_objects: Dict[int, Document] = {}

        # 计算向量检索的 RRF 贡献
        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # 计算 BM25 检索的 RRF 贡献
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # 按总分降序排列
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for doc_id, score in sorted_items:
            doc = doc_objects[doc_id]
            doc.metadata["rrf_score"] = score
            result.append(doc)

        return result


# ============================================================================
# 第二部分: 查询重写 + 上下文构建
# ============================================================================

class ContextBuilder:
    """
    查询重写与上下文构建器

    功能:
        - query_rewrite: 通过 LLM 判断查询是否模糊, 自动重写以提高检索命中率
        - build_context: 将检索到的 Document 列表格式化为结构清晰的文本

    说明:
        - 查询重写依赖外部 LLM (本实现使用 Ollama + ChatOllama)
        - 上下文构建是纯文本拼接, 无外部依赖
    """

    def __init__(self):
        self.llm = None

    # ------------------------------------------------------------------
    # LLM 初始化 (可选, 仅在需要查询重写时调用)
    # ------------------------------------------------------------------

    def setup_llm(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        初始化 LLM (使用 Ollama)

        注意: 需安装 langchain-ollama, 且确保 Ollama 服务已启动

        Args:
            model_name:  Ollama 模型名, 如 "qwen2.5:7b", "llama3.1:8b"
            base_url:    Ollama 服务地址
            temperature: 生成温度 (越低越确定)
            max_tokens:  最大 token 数
        """
        from langchain_ollama import ChatOllama

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )
        logger.info(f"LLM 初始化完成: {model_name} @ {base_url}")

    # ------------------------------------------------------------------
    # 查询重写
    # ------------------------------------------------------------------

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写: 判断是否模糊, 是则重写, 否则原样返回

        策略:
            - 具体明确的问题 → 直接返回原文 (如 "闯红灯扣几分")
            - 模糊宽泛的问题 → 增加相关术语后重写 (如 "交通规则" → "交通安全规定")

        Args:
            query: 原始用户查询

        Returns:
            重写后的查询文本 (如果 LLM 调用失败, 安全回退到原查询)
        """
        if not self.llm:
            logger.warning("LLM 未初始化, 跳过查询重写, 直接返回原查询")
            return query

        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高知识检索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体问题：如"闯红灯扣几分"、"车速超过多少算超速"
   - 明确的法规询问：如"实线变道怎么处罚"、"酒后驾驶标准"
   - 具体参数询问：如"这辆车的续航多少"、"电池容量"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"交通规则"、"车辆参数"
   - 口语化表达：如"有什么规定"、"参数怎么样"

重写原则：
- 保持原意不变
- 增加相关术语
- 保持简洁性

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"],
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            rewritten = chain.invoke(query).strip()
            if rewritten != query:
                logger.info(f"查询已重写: '{query}' → '{rewritten}'")
            else:
                logger.info(f"查询无需重写: '{query}'")
            return rewritten
        except Exception as e:
            logger.error(f"查询重写失败: {e}, 使用原查询")
            return query

    # ------------------------------------------------------------------
    # 上下文构建
    # ------------------------------------------------------------------

    @staticmethod
    def build_context(docs: List[Document], max_length: int = 2000) -> str:
        """
        将检索到的文档块格式化为结构化的上下文字符串

        输出格式:
            【文档 1】 文件名 | 来源: path | 相关性: 0.xxxx
            文档内容...

            ==================================================
            【文档 2】 ...

        Args:
            docs:        检索到的 Document 列表
            max_length:  上下文最大字符数 (达到截断)

        Returns:
            格式化的上下文字符串, 无结果时返回 "暂无相关知识。"
        """
        if not docs:
            return "暂无相关知识。"

        parts = []
        length = 0
        separator = "\n" + "=" * 50 + "\n"

        for i, doc in enumerate(docs, 1):
            # 构建元数据行
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


# ============================================================================
# 使用示例 (if __name__ == "__main__")
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 请替换以下两个变量为你项目中的实际对象:
    #   your_vectorstore  = 你的向量存储实例
    #   your_chunks       = 你的文档块列表 (List[Document])

    print("=" * 60)
    print("RAG 检索与生成参考实现")
    print("=" * 60)
    print()
    print("将本文件集成到你的项目后, 使用方式如下:")
    print()
    print("  from rag_retrieval_generation import RetrievalOptimizer, ContextBuilder")
    print()
    print("  # 1. 混合检索")
    print("  optimizer = RetrievalOptimizer(your_vectorstore, your_chunks)")
    print('  docs = optimizer.hybrid_search("查询文本", top_k=3)')
    print()
    print("  # 2. 上下文构建")
    print('  context = ContextBuilder.build_context(docs, max_length=2000)')
    print("  print(context)")
    print()
    print("  # 3. (可选) 查询重写")
    print('  builder = ContextBuilder()')
    print('  builder.setup_llm(model_name="qwen2.5:7b")')
    print('  rewritten = builder.query_rewrite("交通规则")')
    print()
