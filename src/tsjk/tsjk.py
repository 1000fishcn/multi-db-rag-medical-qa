import pandas as pd
from neo4j import GraphDatabase, exceptions
from dotenv import load_dotenv
import os

# ===================== 1. 加载.env文件配置 =====================

load_dotenv('../../.env')

# 从.env读取配置（自动匹配.env里的变量名）
class Neo4jConfig:
    # Neo4j连接配置
    NEO4J_URI = os.getenv("NEO4J_URI")          # 读取.env里的NEO4J_URI
    NEO4J_USER = os.getenv("NEO4J_USER")        # 读取.env里的NEO4J_USER
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")# 读取.env里的NEO4J_PASSWORD
    DATABASE_NAME = os.getenv("NEO4J_DATABASE") # 读取.env里的NEO4J_DATABASE

    # CSV文件路径
    CSV_PATH = os.getenv("../data/disease.csv")

    # CSV列名映射
    HEAD_COL = os.getenv("CSV_HEAD_COL")
    RELATION_COL = os.getenv("CSV_RELATION_COL")
    TAIL_COL = os.getenv("CSV_TAIL_COL")


# ===================== 2. CSV数据读取与预处理 =====================
def load_and_clean_csv(csv_path, head_col, relation_col, tail_col):
    """读取CSV三元组数据，处理空值、重复值"""
    print(f"\n正在读取CSV文件：{csv_path}")
    # 兼容不同编码（utf-8/gbk）
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")

    # 检查必要列
    required_cols = [head_col, relation_col, tail_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少列：{missing_cols}，请核对.env里的列名配置！")

    # 预处理
    df = df.dropna(subset=required_cols)  # 删除空值行
    df = df.drop_duplicates(subset=required_cols)  # 删除重复三元组
    df = df.reset_index(drop=True)

    print(f"CSV预处理完成：共 {len(df)} 条有效医学三元组")
    return df

# ===================== 3. Neo4j连接与知识图谱构建 =====================
class MedicalKGBuilder:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, db_name):
        """初始化Neo4j连接（从.env读取配置）"""
        try:
            self.driver = GraphDatabase.driver(
                uri=neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            self.db_name = db_name
            self.driver.verify_connectivity()  # 测试连接
            print("✅ Neo4j数据库连接成功！")
        except exceptions.AuthError:
            raise PermissionError("❌ Neo4j用户名/密码错误（请检查.env里的配置）！")
        except exceptions.ServiceUnavailable:
            raise ConnectionError("❌ Neo4j未启动/地址错误（请检查.env里的NEO4J_URI）！")

    def close(self):
        """关闭Neo4j连接"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j连接已关闭")

    def create_medical_kg(self, df, head_col, relation_col, tail_col):
        """批量构建医学知识图谱（节点+关系）"""
        print(f"\n开始构建知识图谱（数据库：{self.db_name}）...")
        with self.driver.session(database=self.db_name) as session:
            total_rows = len(df)
            batch_size = 1000  # 批量提交，提升效率

            for batch_idx in range(0, total_rows, batch_size):
                batch_df = df.iloc[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1

                # Cypher：MERGE避免重复创建节点/关系
                cypher = """
                UNWIND $triples AS triple
                MERGE (h:Entity {name: triple.head})
                MERGE (t:Entity {name: triple.tail})
                MERGE (h)-[r:RELATION {type: triple.relation}]->(t)
                RETURN count(r) AS created_relations
                """

                # 准备批量数据
                triples = batch_df.apply(
                    lambda row: {
                        "head": str(row[head_col]),
                        "relation": str(row[relation_col]),
                        "tail": str(row[tail_col])
                    },
                    axis=1
                ).tolist()

                # 执行Cypher
                try:
                    result = session.run(cypher, triples=triples)
                    processed = result.single()["created_relations"]
                    print(f"✅ 批次 {batch_num}：处理 {processed} 条三元组")
                except exceptions.ClientError as e:
                    print(f"❌ 批次 {batch_num} 失败：{str(e)[:100]}")

        print(f"\n🎉 知识图谱构建完成！总计处理 {total_rows} 条三元组")
        print("👉 查看方式：Neo4j Browser执行 MATCH (n) RETURN n LIMIT 100")

# ===================== 4. 主函数：一键执行 =====================
def main():
    try:

        # 步骤2：读取CSV数据
        df = load_and_clean_csv(
            csv_path=Neo4jConfig.CSV_PATH,
            head_col=Neo4jConfig.HEAD_COL,
            relation_col=Neo4jConfig.RELATION_COL,
            tail_col=Neo4jConfig.TAIL_COL
        )

        # 步骤3：初始化Neo4j并构建图谱
        kg_builder = MedicalKGBuilder(
            neo4j_uri=Neo4jConfig.NEO4J_URI,
            neo4j_user=Neo4jConfig.NEO4J_USER,
            neo4j_password=Neo4jConfig.NEO4J_PASSWORD,
            db_name=Neo4jConfig.DATABASE_NAME
        )

        # 步骤4：构建知识图谱
        kg_builder.create_medical_kg(
            df=df,
            head_col=Neo4jConfig.HEAD_COL,
            relation_col=Neo4jConfig.RELATION_COL,
            tail_col=Neo4jConfig.TAIL_COL
        )

    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
    finally:
        # 确保连接关闭
        try:
            kg_builder.close()
        except:
            pass

if __name__ == "__main__":
    main()