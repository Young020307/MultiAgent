#!/usr/bin/env python3
# rag/build_index.py
"""构建/重建 ChromaDB 向量索引。"""

import os
import shutil

from document_loader import load_documents_from_folder, split_documents
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

COLLECTION_NAME = "customer_service_knowledge"


def rebuild_index(
    docs_dir: str = "./knowledge",
    persist_dir: str = "./chroma_db",
    embedding_model: str = "lrs33/bce-embedding-base_v1:latest",
    ollama_base_url: str = "http://127.0.0.1:11434",
):
    """构建/重建向量索引。每次重建会清空旧数据。"""
    # 清理旧索引，避免追加重复
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

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
        collection_name=COLLECTION_NAME,
    )
    print(f"索引构建完成，持久化目录：{persist_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--docs-dir", default="./knowledge", help="知识文档目录")
    parser.add_argument("--persist-dir", default="./chroma_db", help="ChromaDB 持久化目录")
    parser.add_argument("--embedding-model", default="lrs33/bce-embedding-base_v1:latest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    rebuild_index(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        ollama_base_url=args.ollama_url,
    )
