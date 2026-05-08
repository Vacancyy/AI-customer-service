"""
生成最终QA知识库脚本

功能：
1. 读取Excel标准问答（117条）
2. 读取数据库AI分析问答（151条）
3. 合并相似问题，标准化问题表述
4. 建立一级分类+二级分类结构
5. 导出最终知识库JSON和更新数据库

使用方式：
  python scripts/generate_final_qa.py
"""

import os
import re
import json
import pymysql
import pandas as pd
from collections import Counter, defaultdict

# 数据库配置
DB_CONFIG = {
    'host': 'REMOVED_DB_HOST',
    'port': 3308,
    'user': 'REMOVED_DB_USER',
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': 'ai_customer_service',
    'charset': 'utf8mb4'
}

# ==================== 分类体系 ====================

# 一级分类（大类）
PRIMARY_CATEGORIES = {
    '保障范围': ['保障内容', '保障期限', '既往症保障', '除外责任'],
    '理赔流程': ['理赔步骤', '理赔时效', '理赔条件', '理赔渠道'],
    '理赔材料': ['所需材料', '材料要求', '材料提交'],
    '产品信息': ['产品介绍', '投保条件', '续保规则', '费率价格'],
    '退保流程': ['退保条件', '退保流程', '退保金额', '犹豫期'],
    '条款解释': ['条款含义', '名词解释'],
    '其他问题': ['综合问题', '其他'],
}

# 二级分类映射（数据库现有分类 -> 新分类体系）
CATEGORY_MAPPING = {
    '保障范围了解': ('保障范围', '保障内容'),
    '保障范围类': ('保障范围', '保障内容'),
    '产品了解': ('产品信息', '产品介绍'),
    '产品信息类': ('产品信息', '产品介绍'),
    '理赔流程了解': ('理赔流程', '理赔步骤'),
    '理赔流程类': ('理赔流程', '理赔步骤'),
    '理赔材料了解': ('理赔材料', '所需材料'),
    '理赔材料类': ('理赔材料', '所需材料'),
    '退保流程了解': ('退保流程', '退保流程'),
    '退保流程类': ('退保流程', '退保流程'),
    '条款解释': ('条款解释', '条款含义'),
    '费率价格类': ('产品信息', '费率价格'),
    '其他类': ('其他问题', '其他'),
}


# ==================== 问题标准化 ====================

# 问题标准化映射（相似问题 -> 标准问题）
QUESTION_MERGE_RULES = {
    # 理赔流程类
    '理赔怎么申请': '如何申请理赔',
    '理赔流程是什么': '如何申请理赔',
    '怎么理赔': '如何申请理赔',
    '理赔步骤': '如何申请理赔',
    '怎么办理理赔': '如何申请理赔',
    '如何办理理赔': '如何申请理赔',
    '理赔要什么手续': '如何申请理赔',
    '申请理赔的流程': '如何申请理赔',

    # 理赔材料类
    '理赔需要什么材料': '理赔申请需要哪些材料',
    '理赔材料有哪些': '理赔申请需要哪些材料',
    '需要提交什么材料': '理赔申请需要哪些材料',
    '理赔要准备什么': '理赔申请需要哪些材料',
    '理赔材料清单': '理赔申请需要哪些材料',

    # 保障范围类
    '保什么': '这款产品保障范围是什么',
    '保障什么': '这款产品保障范围是什么',
    '保障范围': '这款产品保障范围是什么',
    '保险范围': '这款产品保障范围是什么',
    '保障内容': '这款产品保障范围是什么',
    '保哪些': '这款产品保障范围是什么',

    # 理赔时效类
    '理赔多久能下来': '理赔需要多长时间',
    '理赔时效': '理赔需要多长时间',
    '多久能赔付': '理赔需要多长时间',
    '赔付时间': '理赔需要多长时间',

    # 保费价格类
    '多少钱': '这款产品保费是多少',
    '保费多少': '这款产品保费是多少',
    '价格多少': '这款产品保费是多少',
    '怎么收费': '这款产品保费是多少',

    # 既往症类
    '既往症能不能保': '既往症是否可以保障',
    '有既往症能买吗': '既往症是否可以保障',
    '既往症理赔吗': '既往症是否可以保障',

    # 退保类
    '怎么退保': '如何办理退保',
    '退保流程': '如何办理退保',
    '能不能退保': '如何办理退保',
    '怎么取消保险': '如何办理退保',
}


def extract_keywords(text):
    """从文本中提取关键词"""
    # 业务关键词列表
    keyword_list = [
        '理赔', '保单', '保险', '报销', '材料', '审核',
        '打款', '到账', '投保', '退保', '续保', '变更',
        '保障', '条款', '费用', '金额', '时效', '流程',
        '申请', '查询', '进度', '状态', '条件', '比例',
        '免赔额', '医保', '门诊', '住院', '门特', '赔付',
        '发票', '清单', '出院', '结算', '直赔', '快赔',
        '既往症', '犹豫期', '等待期', '保险费', '保费',
        '保障期', '保险期限', '承保', '赔付比例',
    ]

    found = []
    for kw in keyword_list:
        if kw in text:
            found.append(kw)

    # 返回前5个关键词
    return ','.join(found[:5]) if found else ''


def standardize_question(question):
    """标准化问题表述"""
    if not question:
        return question

    # 清理文本
    q = question.strip()
    q = re.sub(r'[？?！!。.,，]', '', q)

    # 查找合并规则
    for variant, std in QUESTION_MERGE_RULES.items():
        if variant in q.lower() or q.lower() in variant:
            return std

    return question


def map_category(old_category):
    """将旧分类映射到新的一级+二级分类"""
    if old_category in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[old_category]

    # 尝试智能匹配
    for key, value in CATEGORY_MAPPING.items():
        if key in old_category or old_category in key:
            return value

    # 默认归类到其他
    return ('其他问题', '其他')


def clean_answer(answer):
    """清理回答内容"""
    if not answer:
        return ''

    # 清理多余空格和换行
    ans = answer.strip()
    ans = re.sub(r'\n{3,}', '\n\n', ans)
    ans = re.sub(r' {2,}', ' ', ans)

    # 移除开头问候语（保留一个"您好"）
    greetings = ['您好您好', '你好你好', '您好，您好']
    for g in greetings:
        ans = ans.replace(g, '您好')

    return ans


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def read_excel_qa():
    """读取Excel标准问答"""
    df = pd.read_excel('/home/REMOVED_DB_USER/customer-service/docs/customerQA.xlsx')

    qa_list = []
    for idx, row in df.iterrows():
        std_question = row['标准问']
        answer = row['答复']

        # 根据问题内容推断分类
        primary_cat, secondary_cat = infer_category_from_question(std_question)

        qa_list.append({
            'primary_category': primary_cat,
            'secondary_category': secondary_cat,
            'std_question': std_question,
            'keywords': extract_keywords(std_question + ' ' + str(answer)),
            'answer': clean_answer(str(answer)),
            'priority': 1,  # Excel标准问答优先级最高
            'source': 'excel',
        })

    return qa_list


def read_db_qa():
    """读取数据库AI分析问答"""
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT id, category, question_type, std_question, keywords, answer, source
                FROM qa_knowledge
                WHERE source = 'analysis'
            """)
            rows = cursor.fetchall()

            qa_list = []
            for row in rows:
                # 映射分类
                primary_cat, secondary_cat = map_category(row['category'])

                # 标准化问题
                std_q = standardize_question(row['std_question'])

                qa_list.append({
                    'id': row['id'],
                    'primary_category': primary_cat,
                    'secondary_category': secondary_cat,
                    'std_question': std_q,
                    'keywords': extract_keywords(row['std_question'] + ' ' + str(row['answer'])) if not row['keywords'] else row['keywords'],
                    'answer': clean_answer(str(row['answer'])),
                    'priority': 2,  # AI分析问答优先级次之
                    'source': 'analysis',
                })

            return qa_list
    finally:
        conn.close()


def infer_category_from_question(question):
    """根据问题内容推断分类"""
    q = question.lower()

    # 理赔流程
    if any(kw in q for kw in ['理赔流程', '怎么理赔', '如何理赔', '理赔步骤', '申请理赔']):
        return ('理赔流程', '理赔步骤')
    if any(kw in q for kw in ['理赔多久', '理赔时间', '理赔时效', '多久能赔']):
        return ('理赔流程', '理赔时效')
    if any(kw in q for kw in ['理赔条件', '什么情况理赔']):
        return ('理赔流程', '理赔条件')

    # 理赔材料
    if any(kw in q for kw in ['理赔材料', '需要什么材料', '材料清单', '提交材料']):
        return ('理赔材料', '所需材料')

    # 保障范围
    if any(kw in q for kw in ['保障范围', '保什么', '保障什么', '保障内容']):
        return ('保障范围', '保障内容')
    if any(kw in q for kw in ['保障期限', '保障多久', '保险期限']):
        return ('保障范围', '保障期限')
    if any(kw in q for kw in ['既往症', '既往病史']):
        return ('保障范围', '既往症保障')
    if any(kw in q for kw in ['不赔', '除外责任', '什么情况不赔']):
        return ('保障范围', '除外责任')

    # 产品信息
    if any(kw in q for kw in ['产品介绍', '什么是', '产品详情']):
        return ('产品信息', '产品介绍')
    if any(kw in q for kw in ['投保条件', '谁能买', '年龄限制']):
        return ('产品信息', '投保条件')
    if any(kw in q for kw in ['续保', '续保规则']):
        return ('产品信息', '续保规则')
    if any(kw in q for kw in ['保费', '多少钱', '价格', '费率']):
        return ('产品信息', '费率价格')

    # 退保流程
    if any(kw in q for kw in ['退保', '怎么退', '取消保险']):
        return ('退保流程', '退保流程')
    if any(kw in q for kw in ['犹豫期', '犹豫期多久']):
        return ('退保流程', '犹豫期')

    # 条款解释
    if any(kw in q for kw in ['条款', '什么意思', '名词解释']):
        return ('条款解释', '条款含义')

    return ('其他问题', '其他')


def merge_qa_lists(excel_qa, db_qa):
    """合并Excel和数据库问答，去重并标准化"""
    # 使用字典存储，以标准化问题为key
    merged = {}

    # 先添加Excel问答（优先级高）
    for qa in excel_qa:
        key = qa['std_question']
        if key not in merged:
            merged[key] = qa
        else:
            # 已存在，检查是否需要更新分类
            existing = merged[key]
            if qa['primary_category'] != '其他问题':
                merged[key]['primary_category'] = qa['primary_category']
                merged[key]['secondary_category'] = qa['secondary_category']

    # 再添加AI分析问答
    for qa in db_qa:
        key = qa['std_question']
        if key not in merged:
            merged[key] = qa
        else:
            # 已存在，保留优先级高的，但可能更新回答（如果AI回答更详细）
            existing = merged[key]
            if existing['source'] == 'excel' and qa['source'] == 'analysis':
                # Excel优先，但如果AI回答更长可能更详细
                if len(qa['answer']) > len(existing['answer']) * 1.5:
                    # AI回答明显更长，可以补充
                    merged[key]['answer_supplement'] = qa['answer']

    return list(merged.values())


def update_database(qa_list):
    """更新数据库，添加primary_category和secondary_category字段"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 检查是否需要添加新字段
            cursor.execute("DESCRIBE qa_knowledge")
            columns = [row[0] for row in cursor.fetchall()]

            if 'primary_category' not in columns:
                cursor.execute("ALTER TABLE qa_knowledge ADD COLUMN primary_category VARCHAR(50)")
                print("添加 primary_category 字段")

            if 'secondary_category' not in columns:
                cursor.execute("ALTER TABLE qa_knowledge ADD COLUMN secondary_category VARCHAR(50)")
                print("添加 secondary_category 字段")

            # 更新现有记录的分类
            for qa in qa_list:
                if 'id' in qa and qa.get('id'):
                    cursor.execute("""
                        UPDATE qa_knowledge
                        SET primary_category = %s, secondary_category = %s, std_question = %s
                        WHERE id = %s
                    """, (qa['primary_category'], qa['secondary_category'], qa['std_question'], qa['id']))

            # 删除旧数据并插入新数据
            cursor.execute("DELETE FROM qa_knowledge")

            # 插入合并后的数据
            for qa in qa_list:
                cursor.execute("""
                    INSERT INTO qa_knowledge
                    (primary_category, secondary_category, std_question, keywords, answer, priority, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    qa['primary_category'], qa['secondary_category'],
                    qa['std_question'], qa['keywords'], qa['answer'],
                    qa['priority'], qa['source']
                ))

            conn.commit()
            print(f"数据库更新完成，共 {len(qa_list)} 条记录")

    finally:
        conn.close()


def generate_final_qa():
    """生成最终QA知识库"""
    print("=" * 60)
    print("开始生成最终QA知识库")
    print("=" * 60)

    # 1. 读取Excel标准问答
    print("\n1. 读取Excel标准问答...")
    excel_qa = read_excel_qa()
    print(f"   Excel问答: {len(excel_qa)} 条")

    # 2. 读取数据库AI分析问答
    print("\n2. 读取数据库AI分析问答...")
    db_qa = read_db_qa()
    print(f"   AI分析问答: {len(db_qa)} 条")

    # 3. 合并去重
    print("\n3. 合并去重...")
    merged_qa = merge_qa_lists(excel_qa, db_qa)
    print(f"   合并后总数: {len(merged_qa)} 条")

    # 4. 统计分类分布
    print("\n4. 分类统计:")
    primary_counter = Counter(qa['primary_category'] for qa in merged_qa)
    for cat, count in primary_counter.most_common():
        print(f"   {cat}: {count} 条")

    # 5. 导出JSON
    print("\n5. 导出JSON文件...")
    output_dir = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports'
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, '知识库_最终版.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_qa, f, ensure_ascii=False, indent=2)
    print(f"   JSON文件: {json_path}")

    # 6. 导出CSV（方便查看）
    csv_path = os.path.join(output_dir, '知识库_最终版.csv')
    df = pd.DataFrame(merged_qa)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   CSV文件: {csv_path}")

    # 7. 更新数据库
    print("\n6. 更新数据库...")
    update_database(merged_qa)

    print("\n" + "=" * 60)
    print("最终QA知识库生成完成！")
    print("=" * 60)

    return merged_qa


if __name__ == "__main__":
    generate_final_qa()