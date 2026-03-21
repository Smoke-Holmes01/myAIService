import argparse
import logging
from pathlib import Path

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def default_embedding_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建中国古建筑知识库")
    parser.add_argument(
        "--pdf-dir",
        default="/home/yy/ancient_books",
        help="存放 PDF 资料的目录",
    )
    parser.add_argument(
        "--persist-dir",
        default="/home/yy/my_ai_service/vector_db",
        help="Chroma 向量库输出目录",
    )
    parser.add_argument(
        "--embedding-model",
        default="shibing624/text2vec-base-chinese",
        help="中文向量模型名称",
    )
    parser.add_argument(
        "--device",
        default=default_embedding_device(),
        help="向量模型运行设备，例如 cuda 或 cpu",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="分块大小",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="分块重叠大小",
    )
    return parser.parse_args()


def load_documents(pdf_dir: Path):
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF 目录不存在: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"目录中没有 PDF 文件: {pdf_dir}")

    logger.info("开始加载 PDF，共 %s 个文件", len(pdf_files))
    documents = []

    for pdf_file in pdf_files:
        logger.info("加载 PDF: %s", pdf_file.name)
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = pdf_file.name
        documents.extend(pages)
        logger.info("已加载 %s 页", len(pages))

    logger.info("PDF 总页数: %s", len(documents))
    return documents


def build_vectorstore(args: argparse.Namespace) -> None:
    pdf_dir = Path(args.pdf_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(pdf_dir)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("切分完成，共 %s 个文本块", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("开始写入向量库: %s", persist_dir)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    vectorstore.persist()
    logger.info("知识库构建完成")


if __name__ == "__main__":
    build_vectorstore(parse_args())
