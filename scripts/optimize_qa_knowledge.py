"""
优化QA知识库 - 去重合并相似问题

功能：
1. 智能识别相似问题（语义相近）
2. 合并相似问题，保留最佳回答
3. 优先Excel标准回答，补充AI详细回答
4. 输出优化后的知识库
"""

import os
import re
import json
import pymysql
import pandas as pd
from collections import Counter, defaultdict

PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_CONFIG = {
    'host': 'REMOVED_DB_HOST',
    'port': 3308,
    'user': 'REMOVED_DB_USER',
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': 'ai_customer_service',
    'charset': 'utf8mb4'
}

# ==================== 相似问题合并规则 ====================

# 定义相似问题组，每组合并为一个标准问题
SIMILAR_QUESTION_GROUPS = [
    # 理赔材料类 - 合并为一个标准问题
    {
        'standard': '理赔申请需要哪些材料？',
        'variants': [
            '理赔需要准备哪些材料',
            '理赔需要哪些材料',
            '理赔需要提供哪些材料',
            '申请理赔需要准备哪些材料',
            '理赔材料有哪些',
            '理赔材料清单',
            '理赔要什么材料',
            '理赔材料要求',
        ],
    },
    # 免赔额定义类
    {
        'standard': '什么是免赔额？免赔额是多少？',
        'variants': [
            '免赔额是什么意思',
            '什么是免赔额',
            '免赔额的定义',
            '免赔额是什么',
        ],
    },
    # 免赔额金额类
    {
        'standard': '免赔额具体金额是多少？',
        'variants': [
            '免赔额是多少',
            '免赔额的具体金额是多少',
            '南京宁惠保的免赔额是多少',
            '宁惠保是否有免赔额',
            '免赔额是年度累计的吗',
            '免赔额多少钱',
        ],
    },
    # 既往症定义类
    {
        'standard': '什么是既往症？既往症包括哪些疾病？',
        'variants': [
            '什么是既往症',
            '既往症是什么',
            '既往症的定义是什么',
            '既往症和非既往症的定义是什么',
            '既往症包括哪些',
            '既往症怎么认定',
            '什么是既往症?',
        ],
    },
    # 既往症病种类
    {
        'standard': '本产品既往症具体有哪些？',
        'variants': [
            '本产品既往症有哪些',
            '既往症有哪些疾病',
            '既往症病种',
            '六种既往症是什么',
        ],
    },
    # 理赔流程类
    {
        'standard': '如何申请理赔？理赔流程是什么？',
        'variants': [
            '怎么申请理赔',
            '如何申请理赔',
            '这款产品怎么申请理赔',
            '理赔怎么申请',
            '理赔流程是什么',
            '怎么办理理赔',
            '如何办理理赔',
            '理赔步骤',
            '参保南京宁惠保后如何申请理赔',
            '参保"南京宁惠保"后如何申请理赔',
            '南京宁惠保的理赔申请渠道',
            '理赔申请渠道有哪些',
            '如何理赔',
            '怎么理赔',
        ],
    },
    # 理赔门槛类
    {
        'standard': '理赔门槛是多少？超过多少费用可以申请理赔？',
        'variants': [
            '超过多少费用才可以申请理赔',
            '理赔门槛是多少',
            '南京宁惠保的理赔门槛是多少',
            '多少费用可以理赔',
            '理赔起付线是多少',
        ],
    },
    # 住院报销类
    {
        'standard': '住院费用是否可以报销？如何申请？',
        'variants': [
            '住院费用可以报销吗',
            '住院费用可以报销吗?',
            '住院费用是否可以报销',
            '住院费用如何报销',
            '住院费用如何报销?',
            '住院费用如何申请理赔',
            '住院费用怎么报销',
            '住院能报销吗',
        ],
    },
    # 特药材料类
    {
        'standard': '特药理赔需要哪些材料？',
        'variants': [
            '特药理赔需要准备哪些材料',
            '特药理赔需要哪些材料',
            '特药理赔材料',
            '特药理赔要什么材料',
        ],
    },
    # 直赔快赔类
    {
        'standard': '什么是直赔和快赔？有什么区别？',
        'variants': [
            '什么是直赔',
            '什么是快赔',
            '直赔是什么意思',
            '快赔是什么意思',
            '直赔和快赔的区别',
            '快速理赔是什么',
            '传统理赔和快速理赔有什么区别',
        ],
    },
    # 保障范围类
    {
        'standard': '这款产品保障范围是什么？',
        'variants': [
            '保什么',
            '保障什么',
            '保障范围是什么',
            '保险范围',
            '保障内容',
            '保哪些',
            '这款产品保障什么',
            '这款产品保什么',
        ],
    },
    # 赔付比例类
    {
        'standard': '这款产品的赔付比例是多少？',
        'variants': [
            '赔付比例是多少',
            '赔付比例',
            '赔付多少',
            '赔付率是多少',
            '报销比例是多少',
            '报销比例',
        ],
    },
    # 既往症保障类
    {
        'standard': '既往症是否可以保障？既往症理赔吗？',
        'variants': [
            '既往症能不能保',
            '有既往症能买吗',
            '既往症理赔吗',
            '既往症可以理赔吗',
            '既往症能报销吗',
            '既往症赔付吗',
        ],
    },
    # 保费价格类
    {
        'standard': '这款产品保费是多少？',
        'variants': [
            '多少钱',
            '保费多少',
            '价格多少',
            '怎么收费',
            '保费是多少',
            '这个保险多少钱',
            '宁惠保多少钱',
        ],
    },
    # 退保类
    {
        'standard': '如何办理退保？退保流程是什么？',
        'variants': [
            '怎么退保',
            '退保流程',
            '能不能退保',
            '怎么取消保险',
            '如何退保',
            '退保怎么办理',
            '我想退保',
        ],
    },
    # 门诊报销类
    {
        'standard': '门诊费用是否可以报销？',
        'variants': [
            '门诊费用可以报销吗',
            '门诊费用可以赔付吗',
            '门诊统筹可以赔付吗',
            '门诊能报销吗',
            '门诊费用报销吗',
        ],
    },
    # 异地就医类
    {
        'standard': '异地就医是否可以理赔？',
        'variants': [
            '外地就诊的费用是否能申请理赔',
            '异地就诊的费用是否能申请理赔',
            '异地就医理赔吗',
            '外地就医可以理赔吗',
            '异地就医报销吗',
        ],
    },
    # 理赔时效类
    {
        'standard': '理赔需要多长时间？理赔时效是多久？',
        'variants': [
            '理赔多久能下来',
            '理赔时效',
            '多久能赔付',
            '赔付时间',
            '理赔审核多久',
            '理赔多久到账',
        ],
    },
    # 不赔付类
    {
        'standard': '哪些情况不赔付？除外责任是什么？',
        'variants': [
            '哪些情况不赔付',
            '什么情况不赔',
            '不赔付的情况',
            '除外责任',
            '什么是除外责任',
            '责任免除',
        ],
    },
    # 续保类
    {
        'standard': '这款产品可以续保吗？续保规则是什么？',
        'variants': [
            '可以续保吗',
            '续保规则',
            '续保流程',
            '怎么续保',
            '如何续保',
            '续保条件',
        ],
    },
]

# 一级分类体系
PRIMARY_CATEGORIES = {
    '保障范围': ['保障内容', '保障期限', '既往症保障', '除外责任'],
    '理赔流程': ['理赔步骤', '理赔时效', '理赔条件', '理赔渠道'],
    '理赔材料': ['所需材料', '材料要求', '材料提交'],
    '产品信息': ['产品介绍', '投保条件', '续保规则', '费率价格'],
    '退保流程': ['退保条件', '退保流程', '退保金额', '犹豫期'],
    '条款解释': ['条款含义', '名词解释'],
    '其他问题': ['综合问题', '其他'],
}

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


def normalize_question(question):
    """标准化问题：查找相似问题组，返回标准问题"""
    if not question:
        return question

    q = question.strip()
    q_lower = q.lower()

    # 查找相似问题组
    for group in SIMILAR_QUESTION_GROUPS:
        standard = group['standard']
        for variant in group['variants']:
            # 判断是否相似
            if variant.lower() in q_lower or q_lower in variant.lower():
                return standard
            # 或者问题包含变体的主要关键词
            if len(variant) > 5 and variant.lower() in q_lower.replace('?', '').replace('？', '').replace('吗', '').replace('呢', ''):
                return standard

    return q


def extract_keywords(text):
    """提取关键词"""
    keyword_list = [
        '理赔', '保单', '保险', '报销', '材料', '审核',
        '打款', '到账', '投保', '退保', '续保', '变更',
        '保障', '条款', '费用', '金额', '时效', '流程',
        '申请', '查询', '进度', '状态', '条件', '比例',
        '免赔额', '医保', '门诊', '住院', '门特', '赔付',
        '发票', '清单', '出院', '结算', '直赔', '快赔',
        '既往症', '犹豫期', '等待期', '保险费', '保费',
        '保障期', '保险期限', '承保', '赔付比例',
        '特药', '恶性肿瘤', '质子重离子',
    ]
    found = [kw for kw in keyword_list if kw in str(text)]
    return ','.join(found[:5]) if found else ''


def infer_category(question):
    """根据问题推断分类"""
    q = str(question).lower()

    # 理赔流程
    if any(kw in q for kw in ['理赔流程', '怎么理赔', '如何理赔', '理赔步骤', '申请理赔', '理赔渠道', '理赔方式', '直赔', '快赔']):
        return ('理赔流程', '理赔步骤')
    if any(kw in q for kw in ['理赔多久', '理赔时间', '理赔时效', '多久能赔', '理赔审核', '理赔到账']):
        return ('理赔流程', '理赔时效')
    if any(kw in q for kw in ['理赔条件', '什么情况理赔', '理赔门槛', '起付线', '超过多少费用']):
        return ('理赔流程', '理赔条件')

    # 理赔材料
    if any(kw in q for kw in ['理赔材料', '需要什么材料', '材料清单', '提交材料', '特药材料']):
        return ('理赔材料', '所需材料')

    # 保障范围
    if any(kw in q for kw in ['保障范围', '保什么', '保障什么', '保障内容', '保险范围']):
        return ('保障范围', '保障内容')
    if any(kw in q for kw in ['保障期限', '保障多久', '保险期限']):
        return ('保障范围', '保障期限')
    if any(kw in q for kw in ['既往症是否', '既往症能不能', '既往症可以', '既往症理赔', '既往症保障']):
        return ('保障范围', '既往症保障')
    if any(kw in q for kw in ['不赔', '除外责任', '什么情况不赔', '责任免除']):
        return ('保障范围', '除外责任')

    # 产品信息
    if any(kw in q for kw in ['产品介绍', '什么是', '产品详情', '宁惠保是什么']):
        return ('产品信息', '产品介绍')
    if any(kw in q for kw in ['投保条件', '谁能买', '年龄限制']):
        return ('产品信息', '投保条件')
    if any(kw in q for kw in ['续保', '续保规则', '续保条件']):
        return ('产品信息', '续保规则')
    if any(kw in q for kw in ['保费', '多少钱', '价格', '费率', '怎么收费']):
        return ('产品信息', '费率价格')
    if any(kw in q for kw in ['赔付比例', '赔付多少', '报销比例']):
        return ('产品信息', '费率价格')

    # 退保流程
    if any(kw in q for kw in ['退保', '怎么退', '取消保险', '退保流程']):
        return ('退保流程', '退保流程')
    if any(kw in q for kw in ['犹豫期', '犹豫期多久']):
        return ('退保流程', '犹豫期')

    # 条款解释
    if any(kw in q for kw in ['条款', '什么意思', '名词解释', '免赔额是什么', '既往症是什么', '什么是免赔额', '什么是既往症']):
        return ('条款解释', '条款含义')

    # 门诊住院报销
    if any(kw in q for kw in ['住院费用', '住院报销', '住院理赔', '住院能报销']):
        return ('保障范围', '保障内容')
    if any(kw in q for kw in ['门诊费用', '门诊报销', '门诊理赔', '门诊统筹']):
        return ('保障范围', '保障内容')
    if any(kw in q for kw in ['异地就医', '外地就医', '异地就诊']):
        return ('理赔流程', '理赔条件')

    return ('其他问题', '其他')


def get_best_answer(answers_with_source):
    """从多个回答中选择最佳回答"""
    if not answers_with_source:
        return ''

    # 优先Excel回答
    excel_answers = [a for a, s in answers_with_source if s == 'excel']
    if excel_answers:
        # 选择最长的Excel回答（通常最完整）
        return max(excel_answers, key=len)

    # 其次选择最长的analysis回答
    analysis_answers = [a for a, s in answers_with_source if s == 'analysis']
    if analysis_answers:
        return max(analysis_answers, key=len)

    return ''


def clean_answer(answer):
    """清理回答"""
    if not answer:
        return ''
    ans = str(answer).strip()
    ans = re.sub(r'\n{3,}', '\n\n', ans)
    ans = re.sub(r' {2,}', ' ', ans)
    return ans


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def load_all_qa():
    """加载所有QA数据"""
    # Excel数据
    df = pd.read_excel(os.path.join(PROJECT_ROOT, 'docs/customerQA.xlsx'))
    excel_qa = []
    for _, row in df.iterrows():
        excel_qa.append({
            'question': row['标准问'],
            'answer': str(row['答复']),
            'source': 'excel'
        })

    # 数据库数据
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("SELECT std_question, answer, source FROM qa_knowledge")
            db_qa = [{
                'question': r['std_question'],
                'answer': r['answer'] or '',
                'source': r['source']
            } for r in cursor.fetchall()]
    finally:
        conn.close()

    return excel_qa + db_qa


def merge_and_optimize(all_qa):
    """合并相似问题，优化回答"""
    # 按标准问题归组
    grouped = defaultdict(list)

    for qa in all_qa:
        std_q = normalize_question(qa['question'])
        grouped[std_q].append((qa['answer'], qa['source']))

    # 生成最终QA列表
    final_qa = []
    for std_question, answers_with_source in grouped.items():
        # 选择最佳回答
        best_answer = get_best_answer(answers_with_source)

        # 推断分类
        primary_cat, secondary_cat = infer_category(std_question)

        # 判断来源
        has_excel = any(s == 'excel' for _, s in answers_with_source)
        source = 'excel' if has_excel else 'analysis'
        priority = 1 if has_excel else 2

        final_qa.append({
            'primary_category': primary_cat,
            'secondary_category': secondary_cat,
            'std_question': std_question,
            'keywords': extract_keywords(std_question + best_answer),
            'answer': clean_answer(best_answer),
            'priority': priority,
            'source': source,
            'original_count': len(answers_with_source),  # 原始问题数量
        })

    return final_qa


def update_database(final_qa):
    """更新数据库"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 清空旧数据
            cursor.execute("DELETE FROM qa_knowledge")

            # 插入新数据
            for qa in final_qa:
                cursor.execute("""
                    INSERT INTO qa_knowledge
                    (primary_category, secondary_category, std_question, keywords, answer, priority, source, frequency)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    qa['primary_category'], qa['secondary_category'],
                    qa['std_question'], qa['keywords'], qa['answer'],
                    qa['priority'], qa['source'], qa['original_count']
                ))

            conn.commit()
            print(f"数据库更新完成: {len(final_qa)} 条")
    finally:
        conn.close()


def generate_optimized_qa():
    """生成优化后的QA知识库"""
    print("=" * 60)
    print("优化QA知识库 - 去重合并相似问题")
    print("=" * 60)

    # 加载所有数据
    print("\n1. 加载所有QA数据...")
    all_qa = load_all_qa()
    excel_count = sum(1 for qa in all_qa if qa['source'] == 'excel')
    analysis_count = sum(1 for qa in all_qa if qa['source'] == 'analysis')
    print(f"   Excel: {excel_count} 条")
    print(f"   AI分析: {analysis_count} 条")
    print(f"   总计: {len(all_qa)} 条")

    # 合并优化
    print("\n2. 合并相似问题...")
    final_qa = merge_and_optimize(all_qa)
    print(f"   合并后: {len(final_qa)} 条")
    print(f"   减少: {len(all_qa) - len(final_qa)} 条重复")

    # 分类统计
    print("\n3. 分类统计:")
    primary_counter = Counter(qa['primary_category'] for qa in final_qa)
    for cat, count in primary_counter.most_common():
        print(f"   {cat}: {count} 条")

    # 显示合并效果
    print("\n4. 合并效果示例（原问题数>3）:")
    merged_examples = sorted([qa for qa in final_qa if qa['original_count'] > 3],
                             key=lambda x: -x['original_count'])[:10]
    for qa in merged_examples:
        print(f"   [{qa['original_count']}个相似问题合并为] {qa['std_question'][:50]}...")

    # 导出文件
    print("\n5. 导出文件...")
    output_dir = os.path.join(PROJECT_ROOT, '05_analyze/reports')
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, '知识库_优化版.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_qa, f, ensure_ascii=False, indent=2)
    print(f"   JSON: {json_path}")

    # CSV
    csv_path = os.path.join(output_dir, '知识库_优化版.csv')
    df = pd.DataFrame(final_qa)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   CSV: {csv_path}")

    # 更新数据库
    print("\n6. 更新数据库...")
    update_database(final_qa)

    print("\n" + "=" * 60)
    print("优化完成！知识库条数: ", len(final_qa))
    print("=" * 60)

    return final_qa


if __name__ == "__main__":
    generate_optimized_qa()