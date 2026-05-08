"""
知识库整合脚本 — 执行两项紧急修改：
1. 给引用旧期数(六期/2026年)的条目加注来源标签
2. 合并真正的重复问题（保留最佳回答，其他改为similar_questions）

合并策略：
- "X费用是否可以报销"这类问题虽然句式相似，但问的是不同费用类型，不合并
- 真正重复的是指问同一件事、只是措辞不同的（如"理赔需要多长时间" vs "理赔立案需要多长时间"）
- 手动定义合并组，避免误合并
"""

import json
import copy
from datetime import datetime

QA_PATH = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports/知识库_优化版.json'
BACKUP_PATH = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports/知识库_优化版_整合前备份.json'
OUTPUT_PATH = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports/知识库_整合版.json'

# ========================================================
# 第一步：定义真正的重复合并组
# 每组: (保留的index, [待合并的index列表])
# 被合并的条目删除，其问题作为保留条目的similar_questions
# ========================================================

MERGE_GROUPS = [
    # --- 理赔时效组：7条合并为1条 ---
    # 保留 [53]，它的回答最完整（已按会议纪要更新过）
    (53, [174, 213, 215, 216, 217, 221]),

    # --- 线下理赔地址组：4+4=8条合并为2条 ---
    # [109/171/175/176] 问"线下理赔地址在哪里"
    # [172/193/202/210] 问"线下理赔网点地址和工作时间"
    # 这是两个相关的不同问题，保留2条
    (109, [171, 175, 176]),
    (172, [193, 202, 210]),

    # --- 理赔流程组：6条合并为1条 ---
    # [125/144] 回答完全相同，[162]最完整
    (162, [125, 126, 144, 161, 206]),

    # --- 申请理赔组：4条合并为1条 ---
    # [114/129/167] 问"如何申请宁惠保理赔"，保留[167]回答最详细
    (167, [114, 118, 129]),

    # --- 理赔范围组：6条合并为1条 ---
    # 保留[151]因为它的回答最全面
    (151, [111, 140, 154, 180, 201]),

    # --- 理赔材料组：3条合并为1条 ---
    (52, [127, 205]),

    # --- 理赔方式组：2条合并 ---
    (116, [165]),

    # --- 退保手续组：2条合并 ---
    (191, [143]),  # 保留有问号的[191]，[143]无问号且回答是错的（答了重复短信）

    # --- 参保人群组：2条合并 ---
    (120, [136]),

    # --- 已有其他保险组：2条合并 ---
    (46, [141]),

    # --- 个人自付自费组：2条合并 ---
    (122, [139]),

    # --- 免赔额共享组：3条合并为2条 ---
    # [124/128]问"如何共享+材料"，[149/189]问"是否可以共享"
    (124, [128]),
    (149, [189]),

    # --- 申请医疗理赔组：2条合并 ---
    (130, [218]),

    # --- 理赔范围是什么组：2条合并 ---
    (134, [211]),

    # --- 非南京户籍组：2条合并 ---
    (187, [158]),

    # --- 省外医保组：2条合并 ---
    (173, [179]),

    # --- 住院费用报销组：2条合并 ---
    (186, [184]),

    # --- 急诊费用组：2条合并 ---
    (156, [157]),

    # --- 基因检测组：2条合并 ---
    (112, [207]),

    # --- 门诊费用报销组：2条合并（[22]有问号、[113]无问号）---
    (22, [113]),

    # --- 普通门诊组：2条合并 ---
    (178, [183]),

    # --- 住院费用理赔组：2条合并 ---
    (185, [195]),

    #注意：以下组不合并，因为它们是不同问题（虽然句式相似）：
    # - "X费用是否可以报销"系列（问的是不同费用类型）
    # - "投保是否有X限制"（年龄vs职业是不同限制）
    # - 保费vs保额（完全不同）
    # - 医保结算单vs异地就医结算单（不完全相同）
    # - 理赔金额计算vs免赔额计算（不同概念）
]

# ========================================================
# 第二步：旧期数标注
# 在回答末尾加注来源标签，避免误导
# ========================================================

PERIOD_TAG = "\n\n（注：以上信息为六期产品信息，仅供参考。具体以七期条款为准。）"

# 需要标注的条目（回答中包含六期/2025/2026等旧期数信息的）
# 排除已经准确标注了"六期"且不需额外标注的
def needs_period_tag(answer):
    """判断回答是否需要加注期数来源标签"""
    # 包含旧期数关键词
    has_old_period = any(kw in answer for kw in ['六期', '2026年', '26年', '25年', '2025年'])
    # 已经有标注的不重复加
    already_tagged = '以上信息为' in answer and '期产品信息' in answer
    # 纯当前期信息（如"今年参保"）不需要标注
    is_current = '今年参保' in answer and '2026' not in answer and '六期' not in answer

    return has_old_period and not already_tagged and not is_current


def main():
    # 读取
    with open(QA_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # 备份
    with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print("原始条目数:", len(qa_data))
    print("备份已保存:", BACKUP_PATH)

    # ===== 执行合并 =====
    # 收集所有要删除的index
    to_remove = set()
    merge_log = []

    for keep_idx, remove_indices in MERGE_GROUPS:
        similar_questions = []
        for rm_idx in remove_indices:
            if rm_idx < len(qa_data):
                similar_questions.append(qa_data[rm_idx]['std_question'])
                to_remove.add(rm_idx)

        # 给保留条目添加similar_questions字段
        if keep_idx < len(qa_data):
            existing_similar = qa_data[keep_idx].get('similar_questions', [])
            qa_data[keep_idx]['similar_questions'] = existing_similar + similar_questions
            merge_log.append({
                'kept': qa_data[keep_idx]['std_question'],
                'merged': similar_questions,
                'index_kept': keep_idx,
                'indices_merged': remove_indices,
            })

    # 删除被合并的条目（从大到小删，避免index偏移）
    sorted_remove = sorted(to_remove, reverse=True)
    for idx in sorted_remove:
        qa_data.pop(idx)

    print("\n合并完成:")
    print("  删除条目数:", len(to_remove))
    print("  合并后条目数:", len(qa_data))
    print()

    # 打印合并详情
    for log in merge_log:
        print("  保留: [%d] %s" % (log['index_kept'], log['kept']))
        for q in log['merged']:
            print("    合并: %s" % q)

    # ===== 执行期数标注 =====
    tag_count = 0
    tag_log = []

    for i, qa in enumerate(qa_data):
        if needs_period_tag(qa['answer']):
            # 针对乱码情况特殊处理
            answer = qa['answer']
            # 修复 "2026公众号年1月1日前" 乱码
            answer = answer.replace('2026公众号年1月1日前', '保单生效前（六期2026年1月1日前）')
            answer = answer.replace('2026年1月1日前', '保单生效前（六期2026年1月1日前）')

            qa['answer'] = answer + PERIOD_TAG
            tag_count += 1
            tag_log.append({
                'index': i,
                'question': qa['std_question'],
            })

    print("\n期数标注完成:")
    print("  标注条目数:", tag_count)
    for log in tag_log:
        print("  [%d] %s" % (log['index'], log['question']))

    # ===== 添加 period 字段（按会议纪要要求）=====
    # 销售期关键词
    sales_keywords = ['参保', '投保', '购买', '保费', '停售', '截止日期', '生效时间', '保障期', '优待', '续保']
    # 理赔期关键词
    claim_keywords = ['理赔', '赔付', '报销', '免赔', '快赔', '直赔', '退保', '退费', '材料']

    for qa in qa_data:
        q = qa['std_question']
        a = qa['answer']
        combined = q + a

        is_sales = any(kw in combined for kw in sales_keywords)
        is_claim = any(kw in combined for kw in claim_keywords)

        if is_sales and is_claim:
            qa['period'] = '通用'
        elif is_sales:
            qa['period'] = '销售期'
        elif is_claim:
            qa['period'] = '理赔期'
        else:
            qa['period'] = '通用'

    # 统计
    from collections import Counter
    period_dist = Counter(qa['period'] for qa in qa_data)
    print("\n时期标签分布:")
    for p, c in period_dist.most_common():
        print("  %s: %d" % (p, c))

    # ===== 保存 =====
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print("\n整合版已保存:", OUTPUT_PATH)
    print("最终条目数:", len(qa_data))

    # ===== 保存变更日志 =====
    changelog = {
        'timestamp': datetime.now().isoformat(),
        'original_count': 222,
        'final_count': len(qa_data),
        'removed_count': len(to_remove),
        'period_tagged_count': tag_count,
        'merge_groups': len(MERGE_GROUPS),
        'merge_details': merge_log,
        'period_tag_details': tag_log,
    }
    changelog_path = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports/知识库整合变更日志.json'
    with open(changelog_path, 'w', encoding='utf-8') as f:
        json.dump(changelog, f, ensure_ascii=False, indent=2)
    print("变更日志已保存:", changelog_path)


if __name__ == '__main__':
    main()
