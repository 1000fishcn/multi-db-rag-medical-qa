import os
from git import Repo
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===================== 1. 路径配置 =====================

GUIDELINES_DOWNLOAD_PATH = "./medical_guidelines_cn"
VECTOR_DB_SAVE_PATH = "./vector_db/medical_guidelines_faiss"

# ===================== 2. 嵌入模型配置=====================
# 中文医疗适配的嵌入模型（m3e-base）
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
# 可选：配置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ===================== 3. 核心逻辑=====================
def load_guidelines_as_whole_docs():
    """加载医疗指南，每篇完整文献作为一个 Document（不切分）"""
    # 1. 自动创建语料库目录
    os.makedirs(GUIDELINES_DOWNLOAD_PATH, exist_ok=True)

    # 2. 从 Git 拉语料库
    if not os.listdir(GUIDELINES_DOWNLOAD_PATH):
        # 替换为你实际的医疗指南 Git 仓库地址
        Repo.clone_from("https://github.com/ChiryuhLii/my-TCM-textbook-website.git", GUIDELINES_DOWNLOAD_PATH)

    # 3. 遍历目录，每篇文献作为一个 Document
    docs = []
    file_count = 0
    for root, _, files in os.walk(GUIDELINES_DOWNLOAD_PATH):
        for file in files:
            # 处理文本文件（以后可以根据实际格式调整，比如 .md/.txt/.pdf）
            if file.endswith((".md", ".txt")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 每篇文献直接生成一个 Document
                docs.append(Document(
                    page_content=content,
                    metadata={"source": file_path, "file_name": file}
                ))
                file_count += 1
    return docs


# ===================== 4. 向量库构建/加载示例 =====================
def build_or_load_vector_db():
    # 1. 加载按文献分的 Document 列表
    docs = load_guidelines_as_whole_docs()
    print(f"已加载 {len(docs)} 篇完整医疗指南文献")

    # 2. 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 3. 加载向量库
    if os.path.exists(VECTOR_DB_SAVE_PATH):
        print("正在加载本地向量库...")
        db = FAISS.load_local(VECTOR_DB_SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("正在构建向量库...")
        db = FAISS.from_documents(docs, embeddings)
        # 自动创建向量库目录并保存
        os.makedirs(os.path.dirname(VECTOR_DB_SAVE_PATH), exist_ok=True)
        db.save_local(VECTOR_DB_SAVE_PATH)
        print(f"向量库已保存到 {VECTOR_DB_SAVE_PATH}")
    return db


# 测试运行
if __name__ == "__main__":
    db = build_or_load_vector_db()
    # 简单测试检索
    query = "高血压的诊疗标准"
    results = db.similarity_search(query, k=3)
    print("\n检索结果示例：")
    for res in results:
        print(f"- 来源文献：{res.metadata['file_name']}")
        print(f"  内容片段：{res.page_content[:100]}...\n")