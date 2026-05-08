import os
"""根据客服会议纪要更新知识库：3项修改"""
import json

PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QA_PATH = os.path.join(PROJECT_ROOT, '05_analyze/reports/知识库_优化版.json')

with open(QA_PATH, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

changes = []

for qa in qa_data:
    # ========== 修改1: 理赔时效 ==========
    if qa['std_question'] == '理赔需要多长时间？':
        old = qa['answer']
        qa['answer'] = '您好，理赔时效如下：\n1. 快赔：3个工作日内完成；\n2. 传统理赔：5-7个工作日内完成；\n3. 如需催案，保险公司将在3个工作日内响应处理。\n结案后会有短信通知，具体以短信内容为准。如遇特殊情况需要进一步核实，将在资料完整之日起30天内完成处理。'
        changes.append(('理赔时效', qa['std_question'], old[:60], qa['answer'][:60]))

    # 合并重复的理赔时效条目，统一口径
    if qa['std_question'] in ['理赔立案需要多长时间', '理赔立案需要多长时间？', '理赔审核需要多长时间', '理赔审核需要多长时间？']:
        old = qa['answer']
        qa['answer'] = '您好，理赔时效如下：\n1. 快赔：3个工作日内完成；\n2. 传统理赔：5-7个工作日内完成立案审核；\n3. 如需催案，保险公司将在3个工作日内响应处理。\n结案后会有短信通知，具体以短信内容为准。'
        changes.append(('理赔时效', qa['std_question'], old[:60], qa['answer'][:60]))

    # ========== 修改2: 客服渠道和服务时间 ==========
    if qa['std_question'] == '人工客服服务时间是多少？':
        old = qa['answer']
        qa['answer'] = '您好，客服渠道及服务时间如下：\n1. 电话咨询：拨打4000040181，服务时间每天9:00-21:00；\n2. 企业微信一对一咨询：服务时间每天9:00-21:00；\n3. 私人客服微信一对一咨询；\n4. 在线咨询：服务时间每天9:00-21:00。\n高峰期客服回复可能有延迟，请耐心等待。'
        changes.append(('客服渠道', qa['std_question'], old[:60], qa['answer'][:60]))

    # ========== 修改3: 责任三补充赔付限额说明 ==========
    if qa['std_question'] == '这款产品保障范围是什么？':
        old = qa['answer']
        # 责任二已有限额说明，责任三需补充
        qa['answer'] = qa['answer'].replace(
            '责任三：医保范围外诊疗项目及医疗服务设施医疗保障。保障期间内，被保险人住院或接受门特、门诊大病治疗发生合理且必须的医保范围外诊疗项目及服务设施等费用。',
            '责任三：医保范围外诊疗项目及医疗服务设施医疗保障。保障期间内，被保险人住院或接受门特、门诊大病治疗发生合理且必须的医保范围外诊疗项目及服务设施等费用（设置赔付限额，责任二、责任三年度总赔付额不超过50万元）。'
        )
        # 去掉原来重复的限额说明
        qa['answer'] = qa['answer'].replace('责任三年度总赔付额不超过50万元。\n责任三：', '责任三：')
        changes.append(('责任三限额', qa['std_question'], '缺少赔付限额', '已补充赔付限额'))

with open(QA_PATH, 'w', encoding='utf-8') as f:
    json.dump(qa_data, f, ensure_ascii=False, indent=2)

print(f'修改完成，共{len(changes)}处变更：\n')
for item_type, question, old, new in changes:
    print(f'[{item_type}] {question}')
    print(f'  旧: {old}')
    print(f'  新: {new}')
    print()
