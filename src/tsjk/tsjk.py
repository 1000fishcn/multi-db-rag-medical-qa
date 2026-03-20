import pandas as pd
from neo4j import GraphDatabase, exceptions
from dotenv import load_dotenv
import os

# ===================== 加载.env文件配置 =====================
# 加载.env文件
load_dotenv('../../.env')


# 核心配置
class Neo4jConfig:
    # Neo4j连接配置（从.env读取）
    NEO4J_URI = os.getenv("NEO4J_URI")  # .env里的NEO4J连接地址
    NEO4J_USER = os.getenv("NEO4J_USER")  # .env里的用户名
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # .env里的密码
    DATABASE_NAME = os.getenv("NEO4J_DATABASE")  # .env里的数据库名

    # 三元组CSV文件路径
    CSV_PATH = "./data/disease_triple.csv"

    # 3. 三元组CSV的列名
    HEAD_COL = "head"
    RELATION_COL = "relation"
    TAIL_COL = "tail"


# ===================== Neo4j连接与知识图谱构建 =====================
class MedicalKGBuilder:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, db_name):
        """初始化Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                uri=neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            self.db_name = db_name
            self.driver.verify_connectivity()  # 测试连接
            print("Neo4j数据库连接成功！")
        except exceptions.AuthError:
            raise PermissionError("Neo4j用户名/密码错误（请检查.env里的配置）！")
        except exceptions.ServiceUnavailable:
            raise ConnectionError("Neo4j未启动/地址错误（请检查.env里的NEO4J_URI）！")
        except Exception as e:
            raise RuntimeError(f"Neo4j连接失败：{str(e)}")

    def close(self):
        """关闭Neo4j连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j连接已关闭")

    def create_medical_kg(self, df, head_col, relation_col, tail_col):
        """批量构建医学知识图谱（节点+关系）"""
        # 校验输入数据
        if df.empty:
            raise ValueError("CSV数据为空！请检查文件路径和文件内容")

        # 校验列名是否存在
        required_cols = [head_col, relation_col, tail_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(head_col)
            raise ValueError(f"CSV缺少必要列：{missing_cols}（请检查列名是否为head/relation/tail）")

        print(f"\n开始构建知识图谱（数据库：{self.db_name}）...")
        print(f"待处理三元组总数：{len(df)}")

        with self.driver.session(database=self.db_name) as session:
            batch_size = 1000  # 批量提交，提升大文件处理效率

            for batch_idx in range(0, len(df), batch_size):
                batch_df = df.iloc[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1

                # Cypher语句：MERGE避免重复创建节点/关系
                cypher = """
UNWIND $triples AS triple
// 1. 头节点：统一创建疾病节点（Disease + Entity标签）
MERGE (h:Disease:Entity {name: trim(triple.head)})

// 2. 尾节点：用CASE WHEN动态设置标签（无需WHERE，Neo4j原生支持）
WITH h, triple
// 第一步：创建基础尾节点（所有尾节点都带Entity标签）
MERGE (t:Entity {name: trim(triple.tail)})
// 第二步：根据relation类型，通过CASE WHEN动态添加精准标签（核心修正）
SET t = t {
  .*,  // 保留原有属性
  labels: labels(t) + CASE 
    WHEN triple.relation = '别名' THEN ['Disease']
    WHEN triple.relation = '发病部位' THEN ['BodyPart']
    WHEN triple.relation = '适用人群' THEN ['Population']
    WHEN triple.relation = '传染性' THEN ['InfectionProperty']
    WHEN triple.relation = '医保属性' THEN ['MedicalInsurance']
    WHEN triple.relation = '所属科室' THEN ['Department']
    WHEN triple.relation = '检查项目' THEN ['CheckItem']
    WHEN triple.relation = '典型症状' THEN ['Symptom']
    WHEN triple.relation = '并发症' THEN ['Complication']
    WHEN triple.relation = '治疗方式' THEN ['Treatment']
    WHEN triple.relation = '常用药物' THEN ['Drug']
    WHEN triple.relation = '疗程' THEN ['TreatmentPeriod']
    WHEN triple.relation = '治愈率' THEN ['CureRate']
    WHEN triple.relation = '费用' THEN ['MedicalCost']
    ELSE []  // 未匹配的关系类型，不添加额外标签
  END
}

// 3. 创建头节点→尾节点的关系（RELATION类型，带具体关系名属性）
MERGE (h)-[r:RELATION {type: trim(triple.relation)}]->(t)

// 4. 返回本次批次处理的关系数（用于代码统计）
RETURN count(r) AS created_relations
        """

                # 准备批量数据（清理首尾空格）
                triples = batch_df.apply(
                    lambda row: {
                        "head": str(row[head_col]).strip(),
                        "relation": str(row[relation_col]).strip(),
                        "tail": str(row[tail_col]).strip()
                    },
                    axis=1
                ).tolist()

                # 执行Cypher语句
                try:
                    result = session.run(cypher, triples=triples)
                    processed = result.single()["created_relations"]
                    print(f"批次 {batch_num}：成功处理 {processed} 条三元组")
                except exceptions.ClientError as e:
                    error_msg = str(e)[:100]  # 截取短错误信息，避免刷屏
                    print(f"批次 {batch_num} 处理失败：{error_msg}")

        print(f"知识图谱构建完成！")
        print("查看方式：打开Neo4j Browser，执行 MATCH (n) RETURN n LIMIT 100")


# ===================== 读取CSV数据函数 =====================
def read_triple_csv(csv_path):
    """读取三元组CSV文件"""
    # 校验文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在！路径：{csv_path}")



    with open(csv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        df = pd.read_csv(f)

    print(f"成功读取CSV文件：{csv_path}")
    print(f"CSV数据行数：{len(df)}")
    return df


# ===================== 主函数=====================
def main():
    kg_builder = None  # 初始化变量，避免finally中报错
    try:
        # 读取三元组CSV数据
        print("开始读取三元组CSV文件...")
        df = read_triple_csv(Neo4jConfig.CSV_PATH)

        # 初始化Neo4j连接
        print("\n开始连接Neo4j数据库...")
        kg_builder = MedicalKGBuilder(
            neo4j_uri=Neo4jConfig.NEO4J_URI,
            neo4j_user=Neo4jConfig.NEO4J_USER,
            neo4j_password=Neo4jConfig.NEO4J_PASSWORD,
            db_name=Neo4jConfig.DATABASE_NAME
        )

        # 构建知识图谱
        kg_builder.create_medical_kg(
            df=df,
            head_col=Neo4jConfig.HEAD_COL,
            relation_col=Neo4jConfig.RELATION_COL,
            tail_col=Neo4jConfig.TAIL_COL
        )

    except FileNotFoundError as e:
        print(f"\n执行失败：{str(e)}（请检查CSV路径是否正确）")
    except PermissionError as e:
        print(f"\n执行失败：{str(e)}（请检查Neo4j用户名/密码）")
    except ConnectionError as e:
        print(f"\n执行失败：{str(e)}（请检查Neo4j是否启动/地址是否正确）")
    except ValueError as e:
        print(f"\n执行失败：{str(e)}")
    except Exception as e:
        print(f"\n执行失败：{str(e)}")
    finally:
        # 确保连接关闭
        if kg_builder:
            kg_builder.close()


if __name__ == "__main__":
    main()