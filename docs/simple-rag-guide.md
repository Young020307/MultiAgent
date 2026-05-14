# 简单 RAG 实现指南

> 从 Service 项目中提取的朴素 RAG（Retrieval-Augmented Generation）实现。
> 全部本地运行，基于 LangChain + ChromaDB + Ollama，无需外部 API。

---

## 架构总览

```
┌──────────────────────────────────────────────────┐
│                   离线索引                         │
│  文档文件夹 → 文档加载器 → 文本分块 → 嵌入 → ChromaDB │
└──────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────┐
│                   在线问答                         │
│  用户问题 → 向量检索(top-k) → 拼接Prompt → LLM生成  │
└──────────────────────────────────────────────────┘
```

---

## 1. 依赖安装

```bash
pip install langchain langchain-community langchain-ollama chromadb pypdf docx2txt unstructured
```

确保已安装并启动 [Ollama](https://ollama.com/)，拉取所需模型：

```bash
ollama pull lrs33/bce-embedding-base_v1:latest   # 嵌入模型
ollama pull qwen2.5:3b                            # 推理模型（可替换）
```

---

## 2. 文档加载器（document_loader.py）

支持 .txt / .md / .pdf / .docx / .doc 格式，按文件类型自适应分块。

```python
import os
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """从文件夹中加载所有支持格式的文档。"""
    all_docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[-1].lower()
            try:
                if ext in [".txt", ".md"]:
                    loader = TextLoader(file_path, encoding="utf-8")
                elif ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                elif ext == ".doc":
                    loader = UnstructuredFileLoader(file_path)
                else:
                    print(f"跳过不支持的文件类型：{file}")
                    continue
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_path
                    doc.metadata["file_name"] = file
                all_docs.extend(docs)
                print(f"已加载：{file} ({len(docs)} 页/段)")
            except Exception as e:
                print(f"加载失败 {file}：{e}")
    return all_docs


def split_documents(docs: List[Document]) -> List[Document]:
    """
    按文档类型自适应分块。
    
    策略说明：
    - PDF：chunk_size=800, overlap=100 — PDF 内容密集，块稍大
    - TXT/MD：chunk_size=500, overlap=50 — 纯文本较稀疏
    - DOCX：chunk_size=600, overlap=80
    """
    all_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        ext = os.path.splitext(source)[-1].lower()

        if ext == ".pdf":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )
        elif ext in [".docx", ".doc"]:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600, chunk_overlap=80,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )
        else:  # .txt, .md 及其他
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )

        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
        print(f"  {doc.metadata.get('file_name', '未知')} -> {len(chunks)} 个块")
    return all_chunks
```

---

## 3. 索引构建工具（build_index.py）

将文档加载 → 分块 → 嵌入 → 写入 ChromaDB。

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
# 导入上面的两个函数
# from document_loader import load_documents_from_folder, split_documents


def rebuild_index(
    docs_dir: str = "./docs",
    persist_dir: str = "./chroma_db",
    embedding_model: str = "lrs33/bce-embedding-base_v1:latest",
    ollama_base_url: str = "http://localhost:11434",
):
    """构建/重建向量索引。"""
    print("加载文档...")
    docs = load_documents_from_folder(docs_dir)
    if not docs:
        print("没有文档，跳过索引构建")
        return

    print(f"文档片段数：{len(docs)}")
    chunks = split_documents(docs)
    print(f"切分后文本块数：{len(chunks)}")

    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_base_url,
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"索引构建完成，持久化目录：{persist_dir}")


if __name__ == "__main__":
    rebuild_index()
```

### 使用方式

```
project/
├── docs/               # 放知识文档（.pdf, .txt, .md, .docx）
│   ├── 退换货政策.md
│   ├── 常见问题.pdf
│   └── 服务协议.txt
├── chroma_db/          # 自动生成，持久化向量库
└── build_index.py      # 上述脚本
```

```bash
python build_index.py
```

---

## 4. RAG 问答引擎（knowledge_qa.py）

这是核心：加载向量库 → 检索 → 拼接 Prompt → LLM 生成。

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


class RAG:
    """简单 RAG 问答引擎。"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "lrs33/bce-embedding-base_v1:latest",
        llm_model: str = "qwen2.5:3b",
        ollama_base_url: str = "http://localhost:11434",
        top_k: int = 3,
    ):
        print("初始化嵌入模型...")
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url,
        )

        print("连接向量库...")
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

        print("初始化 LLM...")
        self.llm = ChatOllama(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0,
        )

        self.qa_prompt = ChatPromptTemplate.from_template("""
            你是一个专业的问答助手。请**仅基于**以下参考资料回答用户问题。
            如果参考资料中没有相关信息，请如实告知用户，不要编造任何内容。

            【参考资料】
            {context}

            【用户问题】
            {question}

            【回答】
            """)

    def answer(self, question: str) -> str:
        """检索并生成回答。"""
        docs = self.retriever.invoke(question)
        if not docs:
            return "抱歉，知识库中没有找到相关信息。"

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        chain = self.qa_prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        print(f"[检索] 命中 {len(docs)} 个相关片段")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.metadata.get('file_name', '未知')} "
                  f"(score: {doc.metadata.get('score', 'N/A')})")
        return response.content.strip()


# 快捷函数
def create_rag(**kwargs) -> RAG:
    return RAG(**kwargs)
```

---

## 5. 完整使用示例

```python
# main.py
from knowledge_qa import RAG

# 初始化（启动时执行一次）
rag = RAG(
    persist_directory="./chroma_db",
    top_k=3,
)

# 问答
while True:
    q = input("\n问题（输入 q 退出）：")
    if q.lower() == "q":
        break
    answer = rag.answer(q)
    print(f"\n回答：{answer}")
```

---

## 6. 项目文件结构

```
your-project/
├── docs/               # 知识文档目录
│   ├── policy.md
│   ├── manual.pdf
│   └── faq.txt
├── chroma_db/          # ChromaDB 持久化目录（自动生成）
├── document_loader.py  # 文档加载与分块
├── build_index.py      # 索引构建脚本
├── knowledge_qa.py     # RAG 问答引擎
└── main.py             # 使用示例
```

---

## 7. 关键参数速查

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `chunk_size` | 每块最大字符数 | 500-800，看文档密度 |
| `chunk_overlap` | 块间重叠字符数 | 50-100 |
| `top_k` | 检索返回的文档块数 | 3-5 |
| `temperature` | LLM 随机性 | 知识问答推荐 0 |
| `embedding_model` | 嵌入模型 | 可用 `nomic-embed-text` 等 |
| `llm_model` | 推理模型 | 可用 `qwen2.5`, `llama3.2` 等 |

---

## 8. 进阶改进方向

当前实现是**朴素 RAG**，如需提升效果可逐步加入：

1. **查询改写** — 用 LLM 将用户问题改写得更利于检索
2. **重排序（Reranker）** — 检索后精排，提升 top-k 质量
3. **多路召回** — 同时用关键词（BM25）和向量检索，混合排序
4. **HyDE** — 先让 LLM 生成假设回答，再用该回答检索
5. **引用标注** — 在回答中标注信息来源的文档名

---

> 此实现对应 Service 项目中的 `agents/document_loader.py`、`tools/build_index.py`、`agents/knowledge.py`。
> 使用了 LangChain + ChromaDB + Ollama 全套本地化方案。
