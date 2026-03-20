import pandas as pd
import os


# 读取原始宽表路径
WIDE_CSV_PATH = "./disease.csv"

# 核心映射：CSV列名 → 知识图谱关系名
COL_TO_RELATION = {
    "alias": "别名",
    "part": "发病部位",
    "age": "适用人群",
    "infection": "传染性",
    "insurance": "医保属性",
    "department": "所属科室",
    "checklist": "检查项目",
    "symptom": "典型症状",
    "complication": "并发症",
    "treatment": "治疗方式",
    "drug": "常用药物",
    "period": "疗程",
    "rate": "治愈率",
    "money": "费用"
}



# ===================== 核心处理函数 =====================
def process_wide_csv_to_triple():
    """
    处理逻辑：
    1. 读取疾病宽表
    2. 一行疾病 → 多行三元组（拆分多值）
    3. 保存到原CSV同目录，命名为「原文件名_triple.csv」
    """

    # 读取宽表

    with open(WIDE_CSV_PATH, 'r', encoding='gbk', errors='replace') as f:
        # 从已打开的文件对象读取CSV
        df_wide = pd.read_csv(f)


    # 遍历每行，生成三元组
    triples = []
    total_rows = len(df_wide)
    print(f"开始处理 {total_rows} 条疾病数据...")

    for idx, row in df_wide.iterrows():
        # 头实体：疾病名（去空格、空值）
        head = str(row["name"]).strip()
        # 跳过不存在的行
        if not head or head == "nan":
            continue

        # 遍历所有属性列，生成三元组
        for col, relation in COL_TO_RELATION.items():
            # 跳过不存在的列
            if col not in df_wide.columns:
                continue

            # 尾实体原始值（去空格、空值）
            tail = str(row[col]).strip()
            if not tail or tail == "nan":
                continue

            # 每个拆分后的尾实体生成一条三元组
            triples.append({
                    "head": head,  # 头实体：疾病名
                    "relation": relation,  # 关系：列名对应的友好名称
                    "tail": tail  # 尾实体：属性值
                })

    # 处理三元组：去重、重置索引
    df_triple = pd.DataFrame(triples).drop_duplicates().reset_index(drop=True)


    # 保存到原CSV同目录（自动拼接输出路径）
    # 拆分原路径：目录 + 文件名
    wide_dir = os.path.dirname(WIDE_CSV_PATH)  # 原文件目录
    wide_filename = os.path.basename(WIDE_CSV_PATH)  # 原文件名
    # 新文件名：disease_wide_triple.csv
    triple_filename = os.path.splitext(wide_filename)[0] + "_triple.csv"
    triple_path = os.path.join(wide_dir, triple_filename)  # 输出路径=原目录+新文件名

    # 保存（utf-8-sig避免Excel打开乱码）
    df_triple.to_csv(triple_path, index=False, encoding="utf-8-sig")

    return df_triple


# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        process_wide_csv_to_triple()
    except Exception as e:
        print(f"\n处理失败：{str(e)}")