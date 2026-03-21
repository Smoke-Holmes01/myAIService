from rag_retriever import AncientArchitectureRAG

# 初始化RAG
rag = AncientArchitectureRAG()

# 测试检索
test_questions = [
    "斗拱",
    "营造法式",
    "佛光寺",
    "材分制"
]

for q in test_questions:
    print(f"\n🔍 问题: {q}")
    print("="*50)
    context = rag.get_context(q)
    if context:
        print(f"✅ 找到相关段落:\n{context[:500]}...")
    else:
        print("❌ 没有找到相关内容")