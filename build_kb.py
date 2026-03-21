from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

print("🚀 开始构建古建筑知识库...")

# ===== 1. 加载PDF =====
pdf_dir = "/home/yy/ancient_books"
print(f"📂 从 {pdf_dir} 加载PDF...")

loaders = []
for file in os.listdir(pdf_dir):
    if file.endswith('.pdf'):
        print(f"   发现: {file}")
        loaders.append(PyPDFLoader(os.path.join(pdf_dir, file)))

documents = []
for loader in loaders:
    docs = loader.load()
    print(f"   加载了 {len(docs)} 页")
    documents.extend(docs)

print(f"✅ 共加载 {len(documents)} 个页面片段")

# ===== 2. 切分文档 =====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"✅ 切分成 {len(chunks)} 个文本块")

# ===== 3. 创建向量库 =====
print("🧠 生成向量嵌入...")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="/home/yy/my_ai_service/vector_db"
)

vectorstore.persist()
print(f"✅ 知识库已保存到 /home/yy/my_ai_service/vector_db")
print(f"🎉 完成！共 {len(chunks)} 个知识片段")