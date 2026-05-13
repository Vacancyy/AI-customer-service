"""
知识库更新脚本 - 提供标准化的知识库维护流程
用法:
  python update_knowledge.py add --question "问题" --answer "回答"
  python update_knowledge.py validate
  python update_knowledge.py rebuild
"""

import json
import argparse
import os
import sys

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QA_JSON_PATH = os.path.join(PROJECT_ROOT, '05_analyze/reports/知识库_优化版.json')

# 分类关键词映射（自动推断一级分类）
CATEGORY_KEYWORDS = {
    "产品信息": ["保费", "保额", "保障期", "生效", "投保", "购买", "续保", "犹豫期", "等待期", "保障责任"],
    "理赔流程": ["理赔", "报案", "申请", "流程", "如何理赔", "理赔时间", "理赔方式", "直赔", "快赔"],
    "保障范围": ["保障", "报销", "范围", "病种", "门诊", "住院", "手术", "特药", "双通道"],
    "理赔材料": ["材料", "资料", "发票", "结算单", "出院记录", "病历", "诊断证明"],
    "条款解释": ["条款", "免赔额", "起付线", "既往症", "责任免除", "免责", "赔付比例"],
    "退保流程": ["退保", "退款", "取消", "解除合同"],
    "其他问题": [],
}


def load_knowledge():
    """加载知识库JSON"""
    if not os.path.exists(QA_JSON_PATH):
        print(f"错误: 知识库文件不存在: {QA_JSON_PATH}")
        sys.exit(1)
    with open(QA_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_knowledge(data):
    """保存知识库JSON"""
    with open(QA_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {QA_JSON_PATH} ({len(data)} 条)")


def infer_category(question):
    """根据问题关键词推断一级分类"""
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in question:
                return category
    return "其他问题"


def infer_secondary_category(question, primary_category):
    """推断二级分类"""
    if primary_category == "理赔流程":
        if "直赔" in question or "快赔" in question:
            return "理赔方式"
        elif "时间" in question or "多久" in question:
            return "理赔时效"
        return "理赔申请"
    elif primary_category == "产品信息":
        if "保费" in question:
            return "保费信息"
        elif "保障" in question:
            return "保障责任"
        return "基本信息"
    elif primary_category == "保障范围":
        if "门诊" in question:
            return "门诊保障"
        elif "住院" in question:
            return "住院保障"
        return "保障内容"
    return "其他"


def add_entry(question, answer, category=None, keywords=None):
    """添加新QA条目"""
    data = load_knowledge()

    # 检查重复
    for qa in data:
        if qa['std_question'] == question:
            print(f"警告: 问题已存在")
            return

    # 推断分类
    if not category:
        category = infer_category(question)
    secondary = infer_secondary_category(question, category)

    # 构建关键词
    if not keywords:
        keywords = ','.join([w for w in question if len(w) > 1][:5])

    # 新条目
    new_entry = {
        "primary_category": category,
        "secondary_category": secondary,
        "std_question": question,
        "keywords": keywords,
        "answer": answer,
        "priority": 2,
        "source": "manual_add",
        "original_count": 0
    }

    data.append(new_entry)
    save_knowledge(data)
    print(f"已添加: {question}")
    print(f"  分类: {category}/{secondary}")
    print(f"  当前总数: {len(data)} 条")


def validate_knowledge():
    """校验知识库完整性"""
    data = load_knowledge()
    issues = []

    required_fields = ['std_question', 'answer', 'primary_category']
    for i, qa in enumerate(data):
        for field in required_fields:
            if field not in qa or not qa.get(field):
                issues.append(f"第{i+1}条缺少字段: {field}")

        if qa.get('answer') and len(qa['answer']) < 10:
            issues.append(f"第{i+1}条答案过短")

        for j, other in enumerate(data):
            if i != j and qa.get('std_question') == other.get('std_question'):
                issues.append(f"第{i+1}条与第{j+1}条重复")

    categories = {}
    for qa in data:
        cat = qa.get('primary_category', '未知')
        categories[cat] = categories.get(cat, 0) + 1

    print(f"知识库校验: {len(data)} 条")
    print("分类统计:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} 条")

    if issues:
        print(f"\n发现问题 {len(issues)} 个:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        return False
    else:
        print("\n校验通过")
        return True


def rebuild_vectors():
    """重建向量库"""
    print("正在重建向量库...")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    from ai_qa_system_v2 import ImprovedQASystem
    ImprovedQASystem(rebuild=True)
    print("向量库重建完成")


def main():
    parser = argparse.ArgumentParser(description="知识库更新工具")
    subparsers = parser.add_subparsers(dest='command')

    add_parser = subparsers.add_parser('add', help='添加新QA')
    add_parser.add_argument('--question', required=True)
    add_parser.add_argument('--answer', required=True)
    add_parser.add_argument('--category')
    add_parser.add_argument('--keywords')

    subparsers.add_parser('validate', help='校验知识库')
    subparsers.add_parser('rebuild', help='重建向量库')

    args = parser.parse_args()

    if args.command == 'add':
        add_entry(args.question, args.answer, args.category, args.keywords)
    elif args.command == 'validate':
        validate_knowledge()
    elif args.command == 'rebuild':
        rebuild_vectors()
    else:
        parser.print_help()
        print("\n示例:")
        print("  python update_knowledge.py add --question '七期保费' --answer '基础版99元'")
        print("  python update_knowledge.py validate")
        print("  python update_knowledge.py rebuild")


if __name__ == "__main__":
    main()