import logging
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch


logger = logging.getLogger(__name__)


def _pick_embedding_device() -> str:
    return os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


class AncientArchitectureRAG:
    def __init__(
        self,
        vector_db_path: str | None = None,
        embedding_model_name: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.vector_db_path = vector_db_path or os.getenv(
            "VECTOR_DB_PATH", "/home/yy/my_ai_service/vector_db"
        )
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", "shibing624/text2vec-base-chinese"
        )
        self.top_k = top_k or int(os.getenv("RAG_TOP_K", "3"))
        embedding_device = _pick_embedding_device()

        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(f"知识库目录不存在: {self.vector_db_path}")

        logger.info(
            "加载 RAG 检索器，vector_db=%s, embedding_model=%s, device=%s, top_k=%s",
            self.vector_db_path,
            self.embedding_model_name,
            embedding_device,
            self.top_k,
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def retrieve(self, question: str):
        if not question.strip():
            return []
        return self.retriever.invoke(question)

    def get_context(self, question: str) -> str:
        docs = self.retrieve(question)
        return "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())
