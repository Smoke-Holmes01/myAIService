import argparse

from rag_retriever import AncientArchitectureRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试知识库检索效果")
    parser.add_argument(
        "--vector-db-path",
        default=None,
        help="知识库目录，不传时读取环境变量 VECTOR_DB_PATH",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="要测试的问题，可多次传入",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    questions = args.questions or [
        "斗拱是什么？",
        "《营造法式》的作用是什么？",
        "佛光寺有什么价值？",
        "中国古建筑常见木结构特点有哪些？",
    ]

    rag = AncientArchitectureRAG(vector_db_path=args.vector_db_path)

    for question in questions:
        print(f"\n问题: {question}")
        print("=" * 60)
        context = rag.get_context(question)
        if context:
            print(context[:800])
        else:
            print("没有检索到相关内容。")
