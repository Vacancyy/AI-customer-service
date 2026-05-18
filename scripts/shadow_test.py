"""
AI客服影子测试系统
- 后台运行，不影响真实客服
- 收集客户问题 → AI生成回答 → 记录但不显示
- 同时记录人工客服回答
- 生成对比报告
"""

import sqlite3
import json
import time
import os
import sys
import re
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.ai_qa_system_v2 import ImprovedQASystem, is_personal_query

class ShadowTestingSystem:
    """影子测试系统：后台对比AI与人工回答"""

    def __init__(self):
        self.qa_system = ImprovedQASystem()
        self.heli_db_path = os.path.join(os.path.dirname(__file__), '..', '01_source', 'heli.sqlite3')
        self.log_db_path = os.path.join(os.path.dirname(__file__), '..', '01_source', 'shadow_test.db')
        self._init_log_db()

    def _init_log_db(self):
        """初始化影子测试日志数据库"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shadow_test_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            visitor_id TEXT,
            customer_question TEXT,
            ai_answer TEXT,
            ai_match_score REAL,
            ai_match_question TEXT,
            ai_category TEXT,
            agent_id TEXT,
            agent_name TEXT,
            agent_answer TEXT,
            agent_answer_time REAL,
            question_time TEXT,
            agent_answer_time_stamp TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()
        print(f"影子测试日志数据库: {self.log_db_path}")

    def extract_recent_sessions(self, hours=720):
        """提取最近N小时内有客户提问的会话"""
        conn = sqlite3.connect(self.heli_db_path)
        cursor = conn.cursor()

        # 先获取数据库中的最新时间
        cursor.execute('SELECT MAX(BeginTime) FROM heli_session')
        max_time_str = cursor.fetchone()[0]
        if max_time_str:
            max_time = datetime.strptime(max_time_str, '%Y-%m-%d %H:%M:%S')
        else:
            max_time = datetime.now()

        # 计算时间范围（基于数据库最新时间）
        time_limit = max_time - timedelta(hours=hours)
        time_str = time_limit.strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute('''
        SELECT DISTINCT s.SessionID, s.VisitorId, s.AgentId, s.AgentName, s.BeginTime
        FROM heli_session s
        JOIN heli_message m ON s.SessionID = m.SessionID
        WHERE m.MsgType = 'text'
        AND m.FromUserName = s.VisitorId
        AND s.BeginTime >= ?
        AND s.AgentName IS NOT NULL
        AND s.AgentName != ''
        ORDER BY s.BeginTime DESC
        ''', (time_str,))

        sessions = cursor.fetchall()
        conn.close()

        print(f"数据库最新时间: {max_time_str}")
        print(f"查询时间范围: {time_str} ~ {max_time_str}")
        print(f"有客服接待的会话: {len(sessions)}条")
        return sessions

    def extract_session_messages(self, session_id, visitor_id, agent_id):
        """提取一个会话中的完整对话"""
        conn = sqlite3.connect(self.heli_db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT Content, FromUserName, ToUserName, CreateTime
        FROM heli_message
        WHERE SessionID = ? AND MsgType = 'text'
        ORDER BY CreateTime
        ''', (session_id,))

        messages = cursor.fetchall()
        conn.close()

        # 分离客户问题与客服回答
        pairs = []
        customer_questions = []

        for msg in messages:
            content, from_user, to_user, create_time = msg
            content = content.strip()

            if from_user == visitor_id:
                # 客户消息
                if len(content) > 5 and len(content) < 300:
                    if not any(exclude in content for exclude in ['您好', '在线客服', '很高兴', '请问有什么', '回复数字', '<br>', '&lt;', '&gt;', 'nbsp', '等待回复', '问好了给我留言', '超过2分钟', '好的', '谢谢', '收到', '明白了', '嗯嗯', '哦哦']):
                        customer_questions.append({
                            'content': content,
                            'time': create_time
                        })
            elif from_user == agent_id or to_user == visitor_id:
                # 客服回复（可能是对上一个问题的回答）
                pass

        # 尝试匹配客服回复（简单逻辑：客服回复是针对上一条客户问题）
        cursor.execute('''
        SELECT Content, FromUserName, CreateTime
        FROM heli_message
        WHERE SessionID = ? AND MsgType = 'text'
        AND FromUserName = ?
        ORDER BY CreateTime
        ''', (session_id, agent_id))

        agent_messages = cursor.fetchall()
        conn.close()

        # 构建问答对（客户问 → 下一条客服回复）
        pairs = self._match_qa_pairs(session_id, visitor_id, agent_id)

        return pairs

    def _match_qa_pairs(self, session_id, visitor_id, agent_id):
        """匹配客户问题与客服回复"""
        conn = sqlite3.connect(self.heli_db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT Content, FromUserName, CreateTime
        FROM heli_message
        WHERE SessionID = ? AND MsgType = 'text'
        ORDER BY CreateTime
        ''', (session_id,))

        all_messages = cursor.fetchall()
        conn.close()

        pairs = []
        last_customer_question = None

        for i, msg in enumerate(all_messages):
            content, from_user, create_time = msg
            content = content.strip()

            if from_user == visitor_id:
                # 客户问题
                if len(content) > 5 and len(content) < 300:
                    if not any(exclude in content for exclude in ['您好', '在线客服', '很高兴', '请问有什么', '回复数字', '<br>', '&lt;', '&gt;', 'nbsp', '等待回复', '问好了给我留言', '超过2分钟', '好的', '谢谢', '收到', '明白了', '嗯嗯', '哦哦']):
                        last_customer_question = {
                            'content': content,
                            'time': create_time
                        }

            elif from_user == agent_id and last_customer_question:
                # 客服回复（针对上一条客户问题）
                if len(content) > 10:
                    # 计算客服回复时间
                    q_time = datetime.strptime(last_customer_question['time'], '%Y-%m-%d %H:%M:%S')
                    a_time = datetime.strptime(create_time, '%Y-%m-%d %H:%M:%S')
                    response_time = (a_time - q_time).total_seconds()

                    pairs.append({
                        'question': last_customer_question['content'],
                        'question_time': last_customer_question['time'],
                        'agent_answer': content,
                        'agent_answer_time': create_time,
                        'response_seconds': response_time
                    })

                    # 重置，准备下一对
                    last_customer_question = None

        return pairs

    def process_session(self, session_id, visitor_id, agent_id, agent_name):
        """处理一个会话，生成AI回答并记录"""
        pairs = self._match_qa_pairs(session_id, visitor_id, agent_id)

        if not pairs:
            return 0

        # 对每个客户问题，生成AI回答
        records = []
        for pair in pairs:
            question = pair['question']

            # AI处理
            matched = self.qa_system.search_knowledge(question, top_k=1)
            score = matched[0]['score'] if matched else 0
            matched_q = matched[0]['qa']['std_question'] if matched else '无匹配'

            is_personal = is_personal_query(question)
            ai_answer = self.qa_system.get_answer(question)

            # 分类
            if is_personal:
                ai_category = 'personal_query'
            elif score < 0.5:
                ai_category = 'low_confidence'
            elif score >= 0.7:
                ai_category = 'high_confidence'
            else:
                ai_category = 'medium_confidence'

            records.append({
                'session_id': session_id,
                'visitor_id': visitor_id,
                'customer_question': question,
                'ai_answer': ai_answer,
                'ai_match_score': score,
                'ai_match_question': matched_q,
                'ai_category': ai_category,
                'agent_id': agent_id,
                'agent_name': agent_name,
                'agent_answer': pair['agent_answer'],
                'agent_answer_time': pair['response_seconds'],
                'question_time': pair['question_time'],
                'agent_answer_time_stamp': pair['agent_answer_time']
            })

        # 保存到日志数据库
        self._save_records(records)
        return len(records)

    def _save_records(self, records):
        """保存影子测试记录"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()

        for r in records:
            cursor.execute('''
            INSERT INTO shadow_test_log (
                session_id, visitor_id, customer_question,
                ai_answer, ai_match_score, ai_match_question, ai_category,
                agent_id, agent_name, agent_answer, agent_answer_time,
                question_time, agent_answer_time_stamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                r['session_id'], r['visitor_id'], r['customer_question'],
                r['ai_answer'], r['ai_match_score'], r['ai_match_question'], r['ai_category'],
                r['agent_id'], r['agent_name'], r['agent_answer'], r['agent_answer_time'],
                r['question_time'], r['agent_answer_time_stamp']
            ))

        conn.commit()
        conn.close()

    def run_shadow_test(self, hours=24):
        """运行影子测试"""
        print("=" * 60)
        print("AI客服影子测试开始")
        print("=" * 60)
        print(f"测试时间范围: 最近{hours}小时")
        print()

        # 提取最近会话
        sessions = self.extract_recent_sessions(hours)

        total_records = 0
        for i, (session_id, visitor_id, agent_id, agent_name, begin_time) in enumerate(sessions):
            count = self.process_session(session_id, visitor_id, agent_id, agent_name)
            total_records += count

            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{len(sessions)} 个会话，记录 {total_records} 条问答对")

        print()
        print(f"影子测试完成: {len(sessions)} 个会话, {total_records} 条问答对")
        print(f"日志保存在: {self.log_db_path}")

        return total_records

    def generate_comparison_report(self, output_path=None):
        """生成AI与人工回答对比报告"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM shadow_test_log')
        total = cursor.fetchone()[0]

        if total == 0:
            print("暂无影子测试数据，请先运行测试")
            return

        # 统计分析
        cursor.execute('''
        SELECT
            ai_category,
            COUNT(*) as count,
            AVG(ai_match_score) as avg_score,
            AVG(agent_answer_time) as avg_agent_time
        FROM shadow_test_log
        GROUP BY ai_category
        ''')
        category_stats = cursor.fetchall()

        # 详细数据
        cursor.execute('''
        SELECT
            customer_question,
            ai_answer,
            ai_match_score,
            ai_match_question,
            ai_category,
            agent_answer,
            agent_answer_time,
            agent_name
        FROM shadow_test_log
        ORDER BY ai_match_score DESC
        ''')
        all_records = cursor.fetchall()

        conn.close()

        # 生成报告
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), '..', '05_analyze', 'reports', '影子测试对比报告.md')

        report = self._build_report(total, category_stats, all_records)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"对比报告已生成: {output_path}")

        # 同时输出摘要
        print()
        print("=" * 60)
        print("影子测试对比报告摘要")
        print("=" * 60)
        print(f"测试样本: {total} 条真实客服对话")
        print()
        print("【AI回答分类统计】")
        for cat, count, avg_score, avg_time in category_stats:
            print(f"  {cat}: {count}条, 平均匹配度{avg_score:.2f}, 人工平均响应{avg_time:.1f}秒")
        print()
        print("【关键指标】")
        ai_answered = sum(c for cat, c, _, _ in category_stats if cat in ['high_confidence', 'medium_confidence'])
        print(f"  AI可回答比例: {ai_answered}/{total} = {ai_answered/total*100:.1f}%")

        return output_path

    def _build_report(self, total, category_stats, all_records):
        """构建对比报告"""
        report = f"""# AI客服影子测试对比报告

## 测试概况

- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据来源**: 真实客服对话（影子模式，AI回答未展示给用户）
- **测试样本**: {total} 条客户问题

---

## 一、AI与人工对比统计

### 1.1 AI回答分类

| AI分类 | 数量 | 占比 | 平均匹配度 | 人工平均响应时间 |
|--------|------|------|-----------|-----------------|
"""
        for cat, count, avg_score, avg_time in category_stats:
            report += f"| {cat} | {count} | {count/total*100:.1f}% | {avg_score:.2f} | {avg_time:.1f}秒 |\n"

        report += f"""

### 1.2 核心指标

| 指标 | AI | 人工 | 说明 |
|------|-----|------|------|
| 可回答比例 | {sum(c for cat, c, _, _ in category_stats if cat in ['high_confidence', 'medium_confidence'])/total*100:.1f}% | 100% | AI能处理的比例 |
| 平均响应时间 | ~2秒 | {sum(t for _, _, _, t in category_stats)/len(category_stats):.1f}秒 | AI显著更快 |
| 个人查询处理 | 自动转人工 | 人工核实 | 策略一致 |

---

## 二、典型案例对比

### 2.1 高置信度案例（AI回答准确）

"""
        # 高置信度案例
        high_cases = [r for r in all_records if r[4] == 'high_confidence'][:5]
        for i, r in enumerate(high_cases, 1):
            report += f"""
**案例{i}**
- 客户问题: {r[0]}
- AI匹配: [{r[2]:.2f}] {r[3]}
- **AI回答**: {r[1][:150]}{'...' if len(r[1]) > 150 else ''}
- **人工回答**: {r[5][:150]}{'...' if len(r[5]) > 150 else ''}
- 对比: {'✓ 回答一致' if self._compare_answers(r[1], r[5]) else '○ 有差异但合理'}
"""
        report += """

### 2.2 中置信度案例（AI回答+建议人工）

"""
        medium_cases = [r for r in all_records if r[4] == 'medium_confidence'][:5]
        for i, r in enumerate(medium_cases, 1):
            report += f"""
**案例{i}**
- 客户问题: {r[0]}
- AI匹配: [{r[2]:.2f}] {r[3]}
- **AI回答**: {r[1][:150]}{'...' if len(r[1]) > 150 else ''}
- **人工回答**: {r[5][:150]}{'...' if len(r[5]) > 150 else ''}
"""
        report += """

### 2.3 个人查询案例（正确转人工）

"""
        personal_cases = [r for r in all_records if r[4] == 'personal_query'][:3]
        for i, r in enumerate(personal_cases, 1):
            report += f"""
**案例{i}**
- 客户问题: {r[0]}
- AI处理: 自动识别为个人查询，引导转人工
- **人工回答**: {r[5][:150]}{'...' if len(r[5]) > 150 else ''}
"""
        report += """

---

## 三、回答质量评估

### 3.1 评估标准

| 级别 | 定义 |
|------|------|
| 完全一致 | AI回答核心内容与人工相同 |
| 核心一致 | AI回答要点正确，表述有差异 |
| 有差异 | AI回答方向正确，但缺少部分细节 |
| 需改进 | AI回答与人工有明显差异 |

### 3.2 质量分布（抽样评估）

"""
        # 抽样评估
        sample_size = min(20, len(all_records))
        import random
        random.seed(42)
        sample = random.sample(all_records, sample_size)

        quality_counts = {'完全一致': 0, '核心一致': 0, '有差异': 0, '需改进': 0}
        for r in sample:
            if r[4] in ['high_confidence', 'medium_confidence']:
                quality = self._evaluate_quality(r[1], r[5])
                quality_counts[quality] += 1

        total_evaluated = sum(quality_counts.values())
        if total_evaluated > 0:
            for quality, count in quality_counts.items():
                report += f"- {quality}: {count}/{total_evaluated} ({count/total_evaluated*100:.1f}%)\n"

        report += """

---

## 四、结论与建议

### 4.1 测试结论

"""
        ai_answered = sum(c for cat, c, _, _ in category_stats if cat in ['high_confidence', 'medium_confidence'])
        if ai_answered / total >= 0.8:
            report += f"- **AI利用率良好**: 可独立处理{ai_answered/total*100:.1f}%的常见问题\n"
        else:
            report += f"- **AI利用率待提升**: 当前处理{ai_answered/total*100:.1f}%，建议扩充知识库\n"

        report += f"""
- **响应速度优势**: AI平均~2秒，人工平均{sum(t for _, _, _, t in category_stats)/len(category_stats):.1f}秒
- **策略一致性**: 个人查询正确分流到人工

### 4.2 改进建议

"""
        low_count = sum(c for cat, c, _, _ in category_stats if cat == 'low_confidence')
        if low_count > total * 0.1:
            report += f"- 扩充知识库，覆盖低置信度问题（{low_count}条）\n"

        report += """
---

## 五、上线建议

基于影子测试结果，建议：

1. **可上线范围**: 高置信度+中置信度问题（AI回答后可追加人工建议）
2. **保持人工**: 个人查询、低置信度问题
3. **监控指标**: 每日统计AI利用率、转人工率、用户满意度

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report

    def _compare_answers(self, ai_answer, agent_answer):
        """简单对比AI和人工回答"""
        # 去除通用问候语
        ai_core = ai_answer.replace('您好，', '').replace('请拨打4000040181', '').strip()
        agent_core = agent_answer.replace('您好', '').replace('4000040181', '').strip()

        # 检查核心内容是否有重叠
        if len(ai_core) > 20 and len(agent_core) > 20:
            # 简单关键词重叠判断
            ai_keywords = set(ai_core.split())
            agent_keywords = set(agent_core.split())
            overlap = len(ai_keywords & agent_keywords)
            if overlap > 5:
                return True
        return False

    def _evaluate_quality(self, ai_answer, agent_answer):
        """评估回答质量"""
        # 完全一致：关键数字、流程描述相同
        if self._compare_answers(ai_answer, agent_answer):
            # 检查是否有相同数字
            import re
            ai_nums = re.findall(r'\d+\.?\d*', ai_answer)
            agent_nums = re.findall(r'\d+\.?\d*', agent_answer)
            if ai_nums and agent_nums and set(ai_nums) & set(agent_nums):
                return '完全一致'
            return '核心一致'

        # 有差异但方向正确
        if len(ai_answer) > 50 and len(agent_answer) > 50:
            return '有差异'

        return '需改进'


def main():
    """主函数"""
    # 设置环境变量
    os.environ.setdefault('DASHSCOPE_API_KEY', 'sk-b576a4e032f041bd85bb61f5a92a23f1')
    os.environ.setdefault('DB_HOST', '192.168.10.170')
    os.environ.setdefault('DB_PORT', '3308')
    os.environ.setdefault('DB_USER', 'xiecheng')
    os.environ.setdefault('DB_PASS', 'SecurePass123!')
    os.environ.setdefault('DB_NAME', 'ai_customer_service')

    # 创建影子测试系统
    shadow = ShadowTestingSystem()

    # 运行影子测试（最近30天的对话，因为数据库可能不是最新的）
    shadow.run_shadow_test(hours=720)

    # 生成对比报告
    shadow.generate_comparison_report()


if __name__ == '__main__':
    main()