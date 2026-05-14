# test/test_rag.py
"""Agentic RAG 检索测试 — 单问题检索，验证 JSON 输出结构。"""

import json
import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

from rag import RAG

TEST_QUESTION = "你们的售后政策是啥"

def main():
    rag = RAG(persist_directory="./chroma_db", top_k=3)
    result = rag.search(TEST_QUESTION)
    data = json.loads(result)

    print(f"问题：{TEST_QUESTION}")
    print(f"结果数：{len(data['results'])}")
    print(f"max_score：{data['max_score']:.4f}")
    print(f"low_confidence：{data['low_confidence']}")
    print("=" * 60)
    for i, r in enumerate(data["results"], 1):
        print(f"\n--- 结果 {i} (score: {r['score']:.4f}) ---")
        print(f"来源：{r['metadata'].get('file_name', '?')}")
        print(f"内容({len(r['text'])}字)：{r['text'][:300]}...")


if __name__ == "__main__":
    main()
