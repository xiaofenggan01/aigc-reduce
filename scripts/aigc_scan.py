#!/usr/bin/env python3
"""
AIGC 特征扫描器 — 检测学术论文中的 AI 生成痕迹

基于知网 3.0 / 万方 / PaperPass 的检测原理，扫描以下维度：
  1. 模板句式密度
  2. 句子长度均匀度（突发性）
  3. 嵌套编号模式
  4. 冒号并列结构
  5. 被动语态比例
  6. 段落长度对称性
  7. 模糊表述密度
  8. 标点符号规律性

用法: python aigc_scan.py <file.txt> [--json] [--threshold 0.5]
"""

import re
import sys
import json
import argparse
from collections import Counter
from pathlib import Path


# ─── 模板句式库 ───
TEMPLATE_PATTERNS = [
    # 段落开头模板
    r'^(综上所述[，,])',
    r'^(基于.{2,10}(分析|研究|探讨))',
    r'^(通过.{2,15}(验证|实验|研究|测定))',
    r'^(随着.{2,20}(发展|进步|深入))',
    r'^(近年来[，,])',
    r'^(在.{2,20}(背景下|条件下|过程中))',
    r'^(本研究[旨在对通过])',
    r'^(目前[，,])',
    r'^(当前[，,])',
    r'^(因此[，,])',
    r'^(由此可见[，,])',
    r'^(总而言之[，,])',
    # 过渡词冗余
    r'(此外[，,])',
    r'(另外[，,])',
    r'(与此同时[，,])',
    r'(值得注意的是[，,])',
    r'(需要指出的是[，,])',
    # 模糊表述
    r'(据统计[，,])',
    r'(相关研究表明[，,])',
    r'(一般认为[，,])',
    r'(具有重要的.{2,10}(意义|价值|作用))',
    r'(具有广阔的应用前景)',
    r'(为.{2,20}(提供了|奠定了).{2,10}(基础|依据|参考))',
]

# 被动语态标记
PASSIVE_MARKERS = [
    r'被.{1,15}(测定|检测|验证|确认|证明|发现|计算)',
    r'由.{1,15}(进行|完成|测定|检测|计算)',
    r'经.{1,15}(测定|检测|计算|分析)',
    r'通过.{1,15}(测定|检测|验证|实验|计算)',
    r'采用.{1,15}(进行|测定|检测)',
]

# 嵌套编号模式
NESTED_NUM_PATTERN = re.compile(r'[（(](\d+)[）)]')

# 冒号并列模式
COLON_LIST_PATTERN = re.compile(r'[：:]\s*.+?[；;]\s*.+?[；;]')

# 全角标点
FULLWIDTH_PUNCT = set('，。！？；：、""''（）【】《》')


def load_text(filepath: str) -> str:
    """读取文本文件"""
    path = Path(filepath)
    if not path.exists():
        print(f"错误: 文件不存在 — {filepath}", file=sys.stderr)
        sys.exit(1)

    if path.suffix == '.txt':
        return path.read_text(encoding='utf-8')
    else:
        # 尝试读取纯文本
        return path.read_text(encoding='utf-8')


def split_paragraphs(text: str) -> list[str]:
    """按段落拆分"""
    paras = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paras if p.strip()]


def split_sentences(text: str) -> list[str]:
    """按句子拆分"""
    sentences = re.split(r'[。！？!?\n]', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def count_template_matches(text: str) -> dict:
    """统计模板句式命中"""
    hits = []
    for pattern in TEMPLATE_PATTERNS:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            hits.extend(matches if isinstance(matches[0], str) else [m[0] if isinstance(m, tuple) else m for m in matches])
    return {
        'count': len(hits),
        'density_per_1000': len(hits) / max(len(text), 1) * 1000,
        'matches': hits[:20]  # 最多展示 20 条
    }


def count_passive_markers(text: str) -> dict:
    """统计被动语态"""
    total = 0
    for pattern in PASSIVE_MARKERS:
        total += len(re.findall(pattern, text))
    sentence_count = max(len(split_sentences(text)), 1)
    return {
        'count': total,
        'per_sentence': round(total / sentence_count, 3)
    }


def analyze_burstiness(paragraphs: list[str]) -> dict:
    """分析突发性（句子长度变化程度）"""
    all_sent_lengths = []

    for para in paragraphs:
        sents = split_sentences(para)
        for s in sents:
            all_sent_lengths.append(len(s))

    if len(all_sent_lengths) < 3:
        return {'score': 0, 'avg_len': 0, 'std_len': 0, 'risk': '数据不足'}

    avg_len = sum(all_sent_lengths) / len(all_sent_lengths)
    variance = sum((l - avg_len) ** 2 for l in all_sent_lengths) / len(all_sent_lengths)
    std_len = variance ** 0.5

    # 变异系数 = 标准差 / 均值。AI 文本通常 CV < 0.3
    cv = std_len / max(avg_len, 1)
    if cv < 0.25:
        risk = '高风险 — 句子长度过于均匀，典型 AI 特征'
    elif cv < 0.35:
        risk = '中风险 — 句子变化度偏低'
    elif cv < 0.5:
        risk = '低风险'
    else:
        risk = '正常 — 句子长度变化丰富'

    return {
        'avg_len': round(avg_len, 1),
        'std_len': round(std_len, 1),
        'cv': round(cv, 3),
        'risk': risk
    }


def analyze_para_symmetry(paragraphs: list[str]) -> dict:
    """分析段落长度对称性"""
    para_lens = [len(p) for p in paragraphs]
    if len(para_lens) < 3:
        return {'risk': '段落数不足', 'symmetrical_count': 0}

    # 检测连续 3 段以上长度相近（偏差 < 20%）
    symmetrical_runs = []
    current_run = 0
    for i in range(1, len(para_lens)):
        deviation = abs(para_lens[i] - para_lens[i-1]) / max(para_lens[i-1], 1)
        if deviation < 0.2:
            current_run += 1
        else:
            if current_run >= 2:  # 连续 3 段
                symmetrical_runs.append(current_run + 1)
            current_run = 0
    if current_run >= 2:
        symmetrical_runs.append(current_run + 1)

    return {
        'para_count': len(para_lens),
        'avg_para_len': round(sum(para_lens) / len(para_lens), 1),
        'symmetrical_runs': symmetrical_runs,
        'risk': f'发现 {len(symmetrical_runs)} 处对称段落组（3段以上长度相近）' if symmetrical_runs else '未发现明显对称'
    }


def count_nested_numbers(text: str) -> dict:
    """检测嵌套编号"""
    matches = NESTED_NUM_PATTERN.findall(text)
    return {
        'count': len(matches),
        'risk': f'检测到 {len(matches)} 处编号标记' if len(matches) > 3 else '正常'
    }


def count_colon_lists(text: str) -> dict:
    """检测冒号并列结构"""
    matches = COLON_LIST_PATTERN.findall(text)
    return {
        'count': len(matches),
        'risk': f'检测到 {len(matches)} 处冒号并列结构' if len(matches) > 1 else '正常'
    }


def analyze_punctuation(text: str) -> dict:
    """分析标点规律性"""
    fullwidth_count = sum(1 for c in text if c in FULLWIDTH_PUNCT)
    total_chars = max(len(text), 1)

    # 统计逗号密度
    comma_count = text.count('，') + text.count(',')
    sentence_count = max(len(split_sentences(text)), 1)
    commas_per_sentence = round(comma_count / sentence_count, 2)

    # AI 文本通常逗号密度较高且均匀
    return {
        'fullwidth_punct_ratio': round(fullwidth_count / total_chars * 100, 2),
        'commas_per_sentence': commas_per_sentence,
        'risk': '高密度逗号可能是 AI 特征' if commas_per_sentence > 2.5 else '正常'
    }


def scan(text: str) -> dict:
    """完整扫描"""
    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)

    return {
        'summary': {
            'total_chars': len(text),
            'total_paragraphs': len(paragraphs),
            'total_sentences': len(sentences),
        },
        'template_phrases': count_template_matches(text),
        'passive_voice': count_passive_markers(text),
        'burstiness': analyze_burstiness(paragraphs),
        'para_symmetry': analyze_para_symmetry(paragraphs),
        'nested_numbers': count_nested_numbers(text),
        'colon_lists': count_colon_lists(text),
        'punctuation': analyze_punctuation(text),
    }


def print_report(result: dict):
    """打印人类可读报告"""
    s = result['summary']
    tp = result['template_phrases']
    pv = result['passive_voice']
    b = result['burstiness']
    ps = result['para_symmetry']
    nn = result['nested_numbers']
    cl = result['colon_lists']
    pu = result['punctuation']

    print("=" * 60)
    print("  AIGC 特征扫描报告")
    print("=" * 60)
    print(f"\n文本概况: {s['total_chars']} 字符, {s['total_paragraphs']} 段, {s['total_sentences']} 句")
    print(f"\n{'─' * 40}")
    print("  [1] 模板句式")
    print(f"      命中: {tp['count']} 处 (密度: {tp['density_per_1000']:.2f}/千字)")
    if tp['matches']:
        print(f"      示例: {tp['matches'][:5]}")
    print(f"      风险: {'⚠️  偏高' if tp['count'] > 3 else '✅ 正常'}")

    print(f"\n{'─' * 40}")
    print("  [2] 被动语态")
    print(f"      被动标记: {pv['count']} 处 ({pv['per_sentence']}/句)")
    print(f"      风险: {'⚠️  偏高，建议改主动' if pv['per_sentence'] > 0.3 else '✅ 正常'}")

    print(f"\n{'─' * 40}")
    print("  [3] 句子长度变化（突发性）")
    print(f"      平均句长: {b['avg_len']} 字, 标准差: {b['std_len']}")
    print(f"      变异系数 CV: {b['cv']}")
    print(f"      风险: {b['risk']}")

    print(f"\n{'─' * 40}")
    print("  [4] 段落对称性")
    print(f"      平均段长: {ps['avg_para_len']} 字")
    print(f"      风险: {ps['risk']}")

    print(f"\n{'─' * 40}")
    print("  [5] 嵌套编号")
    print(f"      编号标记: {nn['count']} 处")
    print(f"      风险: {nn['risk']}")

    print(f"\n{'─' * 40}")
    print("  [6] 冒号并列结构")
    print(f"      并列结构: {cl['count']} 处")
    print(f"      风险: {cl['risk']}")

    print(f"\n{'─' * 40}")
    print("  [7] 标点规律")
    print(f"      全角标点占比: {pu['fullwidth_punct_ratio']}%")
    print(f"      每句逗号数: {pu['commas_per_sentence']}")
    print(f"      风险: {pu['risk']}")

    # 综合评分
    risk_count = 0
    if tp['count'] > 3: risk_count += 1
    if pv['per_sentence'] > 0.3: risk_count += 1
    if b.get('cv', 1) < 0.35: risk_count += 1
    if ps.get('symmetrical_runs', []): risk_count += 1
    if nn['count'] > 3: risk_count += 1
    if pu['commas_per_sentence'] > 2.5: risk_count += 1

    print(f"\n{'=' * 60}")
    print(f"  综合风险: {'🔴 高' if risk_count >= 4 else '🟡 中' if risk_count >= 2 else '🟢 低'}")
    print(f"  触发维度: {risk_count}/7")
    print(f"  建议: {'需要多轮深度改写' if risk_count >= 4 else '局部调整即可' if risk_count >= 2 else '风险较低，微调即可'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='AIGC 特征扫描器')
    parser.add_argument('file', help='待扫描的文本文件 (.txt)')
    parser.add_argument('--json', action='store_true', help='输出 JSON 格式')
    parser.add_argument('--threshold', type=float, default=0.5, help='风险阈值（保留）')
    args = parser.parse_args()

    text = load_text(args.file)
    result = scan(text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)


if __name__ == '__main__':
    main()
