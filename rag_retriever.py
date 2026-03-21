from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class AncientArchitectureRAG:
    def __init__(self, vector_db_path="/home/yy/my_ai_service/vector_db"):
        print("🔍 加载RAG检索器...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={'device': 'cuda'}
        )
        
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ RAG就绪")

    def get_context(self, question):
        docs = self.retriever.invoke(question)
        return "\n\n".join([doc.page_content for doc in docs])