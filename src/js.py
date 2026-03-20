import os
import jieba
import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import zhipuai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# ===================== 1. 全局配置与路径 =====================
load_dotenv()

# --- 路径配置  ---
GUIDELINES_DOWNLOAD_PATH = "./zywbxlk"
VECTOR_DB_SAVE_PATH = "./xlsjk/vector_db/medical_guidelines_faiss"

# --- Neo4j 配置 ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- 智谱 API 配置 ---
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4")

# --- 本地模型配置 ---
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 配置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ===================== 2. 核心类：多检索生成器 =====================
class MultiRetrievalGenerator:
    def __init__(self):
        # Neo4j 驱动
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
        )

        # 嵌入模型
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

        # 【核心修改】加载 LangChain 格式的 FAISS 向量库
        self.db, self.text_corpus, self.docstore = self._load_langchain_faiss()

        # 初始化 BM25
        self.bm25 = self._init_bm25()

        # 智谱 API 配置
        zhipuai.api_key = ZHIPU_API_KEY

    def _load_langchain_faiss(self):
        """
        使用官方 HuggingFaceEmbeddings 避免接口不兼容
        """
        try:
            # 1. 使用 LangChain 官方的 Embedding 包装类
            print(f"正在初始化 Embedding 模型...")
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': DEVICE}
            )

            # 2. 使用 LangChain 加载向量库
            print(f"正在从 {VECTOR_DB_SAVE_PATH} 加载向量库...")
            db = FAISS.load_local(
                VECTOR_DB_SAVE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            # 3. 提取纯文本列表 (用于 BM25)
            ordered_documents = []
            for i in range(db.index.ntotal):
                doc_id = db.index_to_docstore_id[i]
                doc = db.docstore.search(doc_id)
                ordered_documents.append(doc)

            text_corpus = [doc.page_content for doc in ordered_documents]

            print(f"成功加载：向量维度 {db.index.d}，文档总数 {len(text_corpus)}")
            return db, text_corpus, ordered_documents

        except Exception as e:
            error_msg = f"""
            加载 FAISS 失败：{str(e)}
            """
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"""
            加载 FAISS 失败：{str(e)}
            请确保已先运行第一段代码生成了向量库文件夹：
            {VECTOR_DB_SAVE_PATH}
            """
            raise ValueError(error_msg)

    def _init_bm25(self):
        """初始化 BM25（中文分词）"""
        tokenized_corpus = [jieba.lcut(text) for text in self.text_corpus]
        return BM25Okapi(tokenized_corpus)

    # ===================== 3. 召回模块 ======================
    def bm25_retrieval(self, query, top_k=10):
        """BM25 文本召回"""
        tokenized_query = jieba.lcut(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = bm25_scores.argsort()[-top_k:][::-1]
        return [(self.text_corpus[idx], bm25_scores[idx]) for idx in top_indices]

    def faiss_retrieval(self, query, top_k=10):
        """FAISS 语义召回 """
        # 使用 LangChain 的 similarity_search_with_score
        # 它返回的是 (Document, score)，这里的 score 通常是距离 (L2)
        docs_and_scores = self.db.similarity_search_with_score(query, k=top_k)

        results = []
        for doc, distance in docs_and_scores:
            # 将距离转换为相似度 (0-1之间，越大越相似)
            similarity = 1 / (1 + distance)
            results.append((doc.page_content, similarity))
        return results

    def neo4j_retrieval(self, query):
        """Neo4j 知识图谱召回 """
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)-[r]->(m)
                    WHERE toLower(n.name) CONTAINS toLower($query) OR toLower(m.name) CONTAINS toLower($query)
                    RETURN n.name as entity1, type(r) as relation, m.name as entity2, n.description as desc1, m.description as desc2
                    LIMIT 5
                    """,
                    query=query
                )
                knowledge = []
                for record in result:
                    knowledge.append(
                        f"实体1：{record['entity1']}，关系：{record['relation']}，实体2：{record['entity2']}，"
                        f"实体1描述：{record['desc1'] or '无'}，实体2描述：{record['desc2'] or '无'}"
                    )
                return "\n".join(knowledge) if knowledge else "未从知识图谱中检索到相关信息"
        except Exception as e:
            return f"Neo4j 检索异常：{str(e)}"

    # ===================== 4. 重排与生成模块 ======================
    def normalize_scores(self, scores):
        if len(scores) == 0: return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: return [1.0 for _ in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def rerank_results(self, query, bm25_results, faiss_results, top_k=5):
        text_score_map = {}
        bm25_texts, bm25_scores = zip(*bm25_results) if bm25_results else ([], [])
        norm_bm25 = self.normalize_scores(bm25_scores)
        for text, score in zip(bm25_texts, norm_bm25):
            text_score_map[text] = {"bm25": score, "faiss": 0.0}

        faiss_texts, faiss_scores = zip(*faiss_results) if faiss_results else ([], [])
        norm_faiss = self.normalize_scores(faiss_scores)
        for text, score in zip(faiss_texts, norm_faiss):
            if text in text_score_map:
                text_score_map[text]["faiss"] = score
            else:
                text_score_map[text] = {"bm25": 0.0, "faiss": score}

        fusion_scores = {t: s["bm25"] * 0.4 + s["faiss"] * 0.6 for t, s in text_score_map.items()}
        sorted_texts = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_texts[:top_k]]

    def generate_answer(self, query, reranked_texts, neo4j_knowledge):
        """调用智谱 GLM-4 生成回答（适配新版 SDK）"""
        prompt = f"""
        你是专业的医疗问答助手，请基于以下信息回答用户问题。
        上下文信息：
        {chr(10).join([f"{i + 1}. {text}" for i, text in enumerate(reranked_texts)])}

        知识图谱信息：
        {neo4j_knowledge}

        用户问题：{query}

        要求：
        1. 仅基于提供的信息回答，不编造内容；
        2. 语言简洁准确，符合医疗常识；
        3. 信息不足时明确说明："根据现有信息无法回答该问题"。
        """

        try:
            # 【关键修改】使用新版 ZhipuAI SDK 写法
            client = zhipuai.ZhipuAI(api_key=ZHIPU_API_KEY)

            response = client.chat.completions.create(
                model=ZHIPU_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            # 返回提取的内容
            return response.choices[0].message.content

        except Exception as e:
            return f"大模型调用失败：{str(e)}"

    def run(self, query, top_k_recall=10, top_k_rerank=5):
        bm25_results = self.bm25_retrieval(query, top_k_recall)
        faiss_results = self.faiss_retrieval(query, top_k_recall)
        neo4j_knowledge = self.neo4j_retrieval(query)
        reranked_texts = self.rerank_results(query, bm25_results, faiss_results, top_k_rerank)
        answer = self.generate_answer(query, reranked_texts, neo4j_knowledge)
        return {"query": query, "reranked_texts": reranked_texts, "final_answer": answer}


# ===================== 5. 测试运行 ======================
if __name__ == "__main__":
    mr_generator = None
    try:
        print("正在初始化系统，请稍候...")
        mr_generator = MultiRetrievalGenerator()
        print("系统初始化完成！")
        print("=" * 50)
        print("请输入你的问题（输入 'quit' 或 'exit' 退出）：")

        while True:
            # 获取用户输入
            user_input = input("\n请输入问题: ").strip()

            # 退出判断
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break

            # 空输入跳过
            if not user_input:
                continue

            print("正在思考中...")
            # 运行主流程
            result = mr_generator.run(user_input)

            # 打印结果
            print("\n" + "=" * 30)
            print("最终回答：")
            print(result["final_answer"])
            print("=" * 30)

    except Exception as e:
        print(f"运行出错：{str(e)}")
    finally:
        # 确保关闭 Neo4j 驱动
        if mr_generator and hasattr(mr_generator, 'neo4j_driver'):
            mr_generator.neo4j_driver.close()
            print("\nNeo4j 驱动已关闭")