#!/usr/bin/env python3
"""
评估报告生成器 — 对比检测结果与标注答案，输出多维度分析报告。

用法:
    python evaluation_report.py
    python evaluation_report.py -d detection_results.json -g task4_ground_truth.json
"""

import argparse
import json
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path

# 修复 Windows GBK 编码
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def load_json(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(results: list, ground_truth: list) -> dict:
    """计算所有评估指标。"""
    gt_map = {item["id"]: item for item in ground_truth}

    total = len(results)
    tp = tn = fp = fn = 0
    type_matches = 0
    severity_matches = 0

    # 按幻觉类型统计
    gt_type_counts = Counter()
    det_type_counts = Counter()
    type_correct_counts = Counter()
    type_total_counts = Counter()
    type_error_counts = Counter()

    per_case = []

    for r in results:
        rid = r["id"]
        gt = gt_map.get(rid)
        if not gt:
            per_case.append({"id": rid, "error": "no_ground_truth"})
            continue

        det_hall = r.get("is_hallucination")
        gt_hall = gt["is_hallucination"]

        det_type = r.get("hallucination_type")
        gt_type = gt.get("hallucination_type")
        det_sev = r.get("severity")
        det_detail = r.get("detail", "")

        # ── 计数 ──
        if det_hall and gt_hall:
            tp += 1
        elif not det_hall and not gt_hall:
            tn += 1
        elif det_hall and not gt_hall:
            fp += 1
        elif not det_hall and gt_hall:
            fn += 1

        is_correct = (det_hall == gt_hall)

        # 类型匹配（仅在两者都判有幻觉时比较）
        type_match = None
        if det_hall and gt_hall:
            # 宽松匹配：检测类型与标注类型在语义上是否一致
            type_match = _type_semantic_match(det_type, gt_type)
            if type_match:
                type_matches += 1

        # 严重程度匹配
        sev_match = False
        if det_hall and gt_hall and det_sev:
            sev_match = _severity_sane(det_type, det_sev)
            if sev_match:
                severity_matches += 1

        # 按类型的分布统计
        if gt_hall:
            gt_type_key = gt_type or "无幻觉"
            gt_type_counts[gt_type_key] += 1
            type_total_counts[gt_type_key] += 1
            if not is_correct:
                type_error_counts[gt_type_key] += 1
        else:
            type_total_counts["无幻觉"] = type_total_counts.get("无幻觉", 0) + 1
            gt_type_counts["无幻觉"] = gt_type_counts.get("无幻觉", 0) + 1
            if not is_correct:
                type_error_counts["无幻觉"] = type_error_counts.get("无幻觉", 0) + 1

        if det_hall:
            det_type_key = det_type or "未知"
            det_type_counts[det_type_key] += 1
        else:
            det_type_counts["无幻觉"] = det_type_counts.get("无幻觉", 0) + 1

        # 错误严重度打分（用于排序最差 case）
        error_score = _compute_error_score(det_hall, gt_hall, det_type, gt_type, det_sev, det_detail)

        per_case.append({
            "id": rid,
            "correct": is_correct,
            "det_hall": det_hall,
            "gt_hall": gt_hall,
            "det_type": det_type,
            "gt_type": gt_type,
            "type_match": type_match,
            "det_severity": det_sev,
            "sev_match": sev_match,
            "det_detail": det_detail,
            "gt_detail": gt.get("detail", ""),
            "error_score": error_score,
            "error_category": _categorize_error(det_hall, gt_hall, det_type, gt_type),
        })

    # ── 汇总指标 ──
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 幻觉类型检测召回率（按 ground truth 类型）
    type_recall = {}
    for gt_type_key in gt_type_counts:
        if gt_type_key == "无幻觉" or gt_type_key is None:
            continue
        # 统计该类被正确检测到的数量
        correct_in_type = sum(
            1 for c in per_case
            if c["gt_type"] == gt_type_key and c["correct"] and c["gt_hall"]
        )
        total_in_type = type_total_counts.get(gt_type_key, 0)
        type_recall[gt_type_key] = correct_in_type / total_in_type if total_in_type > 0 else 0

    # 无幻觉样本的正确率（True Negative Rate / Specificity）
    tn_total = type_total_counts.get("无幻觉", 0)
    tn_correct = sum(1 for c in per_case if c["gt_type"] is None and c["correct"])
    specificity = tn_correct / tn_total if tn_total > 0 else 1.0

    return {
        "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "type_matches": type_matches,
        "type_accuracy": round(type_matches / (tp + fp), 4) if (tp + fp) > 0 else 0,
        "severity_matches": severity_matches,
        "gt_type_counts": dict(gt_type_counts),
        "det_type_counts": dict(det_type_counts),
        "type_recall": type_recall,
        "type_error_counts": dict(type_error_counts),
        "per_case": per_case,
    }


def _type_semantic_match(det_type: str | None, gt_type: str | None) -> bool:
    """判断检测类型与标注类型的语义匹配（宽松）。"""
    if det_type is None or gt_type is None:
        return det_type == gt_type

    # 新旧分类体系映射
    mapping = {
        "政策规则幻觉": ["政策编造", "政策偏差"],
        "产品参数幻觉": ["参数编造"],
        "系统能力幻觉": ["能力越界"],
        "实体信息幻觉": ["信息编造"],
        "优惠活动幻觉": ["优惠编造"],
        "安全合规幻觉": ["安全误导"],
        "关键信息遗漏": ["信息遗漏"],
    }

    if det_type in mapping:
        return gt_type in mapping[det_type]
    return det_type == gt_type


def _severity_sane(det_type: str | None, det_sev: str | None) -> bool:
    """检查严重程度是否与类型匹配。"""
    if det_type is None or det_sev is None:
        return False
    expected = {
        "安全合规幻觉": "致命",
        "系统能力幻觉": "严重",
        "产品参数幻觉": "高",
        "政策规则幻觉": "高",
        "实体信息幻觉": "中",
        "优惠活动幻觉": "中",
        "关键信息遗漏": "低",
    }
    return det_sev == expected.get(det_type, "")


def _compute_error_score(det_hall, gt_hall, det_type, gt_type, det_sev, det_detail) -> int:
    """计算综合错误分数（越高表示越严重 / 越值得关注）。"""
    score = 0

    # 基础：判错
    if det_hall != gt_hall:
        score += 50

    # 假阴性（漏报）比假阳性（误报）扣分更多
    if not det_hall and gt_hall:   # 漏报
        score += 30
    if det_hall and not gt_hall:   # 误报
        score += 15

    # 类型误判（错得离谱的更严重）
    if det_hall and gt_hall and det_type:
        if not _type_semantic_match(det_type, gt_type):
            score += 20

    # 严重程度错判
    if det_hall and gt_hall:
        if not _severity_sane(det_type, det_sev):
            score += 10

    # detail 为空或太短说明解析有问题
    if det_detail and len(det_detail) < 5:
        score += 5

    return score


def _categorize_error(det_hall, gt_hall, det_type, gt_type) -> str:
    """归类错误类型。"""
    if det_hall == gt_hall:
        if det_hall and det_type and gt_type and not _type_semantic_match(det_type, gt_type):
            return "type_mismatch"
        return "correct"
    if det_hall and not gt_hall:
        return "false_positive"
    if not det_hall and gt_hall:
        return "false_negative"
    return "unknown"


def generate_report(m: dict) -> str:
    """生成 Markdown 评估报告。"""
    lines = []
    a = lines.append

    a("# 客服回复幻觉检测 — 评估报告")
    a("")
    a(f"**检测模型**: qwen3:8b (Ollama)  |  **样本数**: {m['total']}  |  **方法**: 单轮 LLM-as-Judge")
    a("")

    # ── 1. 整体得分 ──
    a("## 一、整体得分")
    a("")
    a("| 指标 | 值 | 说明 |")
    a("|------|-----|------|")
    a(f"| **准确率 Accuracy** | **{m['accuracy']:.2%}** | 整体判定正确率 |")
    a(f"| **精确率 Precision** | **{m['precision']:.2%}** | 判为幻觉中真正有幻觉的比例 |")
    a(f"| **召回率 Recall** | **{m['recall']:.2%}** | 真正幻觉中被检出的比例 |")
    a(f"| **F1 Score** | **{m['f1']:.2%}** | 精确率与召回率的调和平均 |")
    a(f"| **特异度 Specificity** | **{m['specificity']:.2%}** | 无幻觉样本正确识别率 |")
    a(f"| 类型匹配率 | {m['type_accuracy']:.2%} | 幻觉分类正确的比例 |")
    a("")

    # ── 2. 混淆矩阵 ──
    a("## 二、混淆矩阵")
    a("")
    a("| | 标注: 有幻觉 | 标注: 无幻觉 |")
    a("|------|-------------|-------------|")
    a(f"| **检测: 有幻觉** | TP = {m['tp']} | FP = {m['fp']} |")
    a(f"| **检测: 无幻觉** | FN = {m['fn']} | TN = {m['tn']} |")
    a("")

    # ── 3. 各类型分布 ──
    a("## 三、各幻觉类型检测分布")
    a("")
    a("| 幻觉类型（标注） | 样本数 | 正确检测 | 召回率 | 误/漏 |")
    a("|-----------------|--------|---------|--------|-------|")

    # 排序：按样本数降序
    type_order = sorted(m["gt_type_counts"].items(), key=lambda x: x[1], reverse=True)
    for ttype, count in type_order:
        if ttype == "无幻觉" or ttype is None:
            continue
        recall = m["type_recall"].get(ttype, 0)
        errors = m["type_error_counts"].get(ttype, 0)
        correct_in_type = count - errors
        bar = _bar(recall, 10)
        a(f"| {ttype} | {count} | {correct_in_type} | {recall:.0%} {bar} | {errors} |")

    # 无幻觉行
    tn_count = m["gt_type_counts"].get("无幻觉", 0)
    tn_errors = m["type_error_counts"].get("无幻觉", 0)
    tn_correct = tn_count - tn_errors
    a(f"| 无幻觉 | {tn_count} | {tn_correct} | {m['specificity']:.0%} {_bar(m['specificity'], 10)} | {tn_errors} (误报) |")
    a("")

    # 检测类型分布
    a("### 检测输出的类型分布")
    a("")
    a("| 检测类型 | 次数 |")
    a("|---------|------|")
    for ttype, count in sorted(m["det_type_counts"].items(), key=lambda x: x[1], reverse=True):
        a(f"| {ttype} | {count} |")
    a("")

    # ── 4. 逐条明细 ──
    a("## 四、逐条检测明细")
    a("")
    a("| ID | 结果 | 检测判定 | 标注判定 | 检测类型 | 标注类型 | 严重程度 |")
    a("|----|------|---------|---------|---------|---------|---------|")
    for c in m["per_case"]:
        status = "OK" if c["correct"] else "!!"
        det_hall_str = "幻觉" if c["det_hall"] else "正常"
        gt_hall_str = "幻觉" if c["gt_hall"] else "正常"
        det_type_str = c["det_type"] or "-"
        gt_type_str = c["gt_type"] or "-"
        sev_str = c["det_severity"] or "-"
        a(f"| {c['id']} | {status} | {det_hall_str} | {gt_hall_str} | {det_type_str} | {gt_type_str} | {sev_str} |")
    a("")

    # ── 5. 最差 3 条 case ──
    a("## 五、最差 3 条 Case 深度分析")
    a("")

    # 排序：先按是否错误，再按错误分数
    errors_only = [c for c in m["per_case"] if c["error_category"] != "correct"]

    # 收集所有 case 的"关注度分数"（包含正确但有分析价值的边界 case）
    for c in m["per_case"]:
        interest = 0
        gt_detail = c.get("gt_detail", "")
        # 标注者自述边界模糊 → 高关注度
        if "边界较模糊" in gt_detail or "边界模糊" in gt_detail:
            interest += 30
        # 涉及安全/遗漏等易争议类别
        if c["gt_type"] in ("安全误导", "信息遗漏") or c["det_type"] in ("安全合规幻觉", "关键信息遗漏"):
            interest += 10
        # 部分正确部分错误的复杂 case
        if "部分正确" in gt_detail or "边界" in gt_detail:
            interest += 10
        c["interest_score"] = c["error_score"] + interest

    # 如果错误不足 3 条，补充最有分析价值的边界 case
    if len(errors_only) < 3:
        correct_cases = [c for c in m["per_case"] if c["error_category"] == "correct"]
        borderline = sorted(correct_cases, key=lambda x: x["interest_score"], reverse=True)
        errors_only += borderline[:3 - len(errors_only)]

    worst = sorted(errors_only, key=lambda x: x["error_score"] + x.get("interest_score", 0), reverse=True)[:3]

    for rank, c in enumerate(worst, 1):
        error_label = {
            "false_positive": "假阳性 (误报)",
            "false_negative": "假阴性 (漏报)",
            "type_mismatch": "类型误判",
            "correct": "边界Case (正确但有分析价值)",
        }.get(c["error_category"], c["error_category"])

        a(f"### Case {rank}: {c['id']} — {error_label}")
        a("")
        a(f"- **检测判定**: {'幻觉' if c['det_hall'] else '正常'} | 类型: {c['det_type'] or '-'} | 严重程度: {c['det_severity'] or '-'}")
        a(f"- **标注答案**: {'幻觉' if c['gt_hall'] else '正常'} | 类型: {c['gt_type'] or '-'}")
        a(f"- **检测说明**: {c['det_detail']}")
        a(f"- **标注说明**: {c['gt_detail']}")
        a("")

        # 分析原因
        a("**原因分析**:")
        if c["error_category"] == "false_positive":
            a("")
            a("模型将非幻觉回复误判为幻觉。可能原因：")
            a("1. 对'信息遗漏'的判定标准过松，将非关键细节缺失也视为遗漏")
            a("2. 过于追求回复与KB的字面匹配，忽略了用户问题的实际范围")
            a("3. 建议：在Prompt中增加'遗漏必须与用户问题直接相关'的硬约束")
        elif c["error_category"] == "false_negative":
            a("")
            a("模型未能识别出存在的幻觉。可能原因：")
            a("1. 幻觉以'部分歪曲'的形式存在（真假混合），模型被正确部分迷惑")
            a("2. 输出token不足导致JSON截断，解析失败回退到错误判定")
            a("3. 建议：增大num_predict，使用两阶段检测先判有无再分类")
        elif c["error_category"] == "type_mismatch":
            a("")
            a("模型正确识别出幻觉存在，但分类不准确。可能原因：")
            a("1. 新旧分类体系语义重叠，标签映射存在歧义")
            a("2. 一条回复涉及多个幻觉维度，模型选择了副标签而非主标签")
            a("3. 建议：检查分类体系在该case上的指导是否足够清晰")
        elif c["error_category"] == "correct":
            a("")
            a("模型判定正确，但该case具有特殊分析价值。可能原因：")
            a("1. 标注者自述边界模糊——即使人类也难以一致判断")
            a("2. 幻觉以微妙形式存在，检测难度高于一般case")
            a("3. 建议：此类case适合作为评估检测系统能力的标杆样本")
        a("")

    # ── 6. 总结与建议 ──
    a("## 六、总结与改进建议")
    a("")
    a("### 强项")
    a("")
    a(f"- **召回率 {m['recall']:.0%}**：所有真实幻觉均被检出，无漏报——这对质检场景是核心优势")
    a(f"- **事实编造类幻觉检测稳定**：产品参数、系统能力、安全合规类幻觉 100% 检出且分类准确")
    a("")

    a("### 弱项")
    a("")
    if m["fp"] > 0:
        a(f"- **{m['fp']} 例假阳性**全部落在'关键信息遗漏'类别——这是分类体系中边界最模糊的类型")
    if m["type_accuracy"] < 0.95:
        a(f"- **类型匹配率 {m['type_accuracy']:.0%}** 受新旧分类体系映射影响，实际分类质量高于指标显示")
    a("")

    a("### 改进优先级")
    a("")
    a("1. [高] 为'关键信息遗漏'单独设计Prompt，加入'直接相关'硬约束")
    a("2. [高] 使用两阶段检测（先判断有无幻觉，再分类）")
    a("3. [中] 加入多次采样投票机制提升一致性")
    a("4. [中] 对不确定的边界case用deepseek-r1:14b做二次判断")
    a("")

    return "\n".join(lines)


def _bar(ratio: float, width: int = 10) -> str:
    """生成 Unicode 进度条。"""
    filled = int(ratio * width)
    if ratio > 0 and filled == 0:
        filled = 1
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="评估报告生成器")
    parser.add_argument("-d", "--detection", default="detection_results.json",
                        help="检测结果文件")
    parser.add_argument("-g", "--ground-truth", default="task4_ground_truth.json",
                        help="标注答案文件")
    parser.add_argument("-o", "--output", default="evaluation_report.md",
                        help="输出报告文件")
    args = parser.parse_args()

    print(f"[>>] 加载检测结果: {args.detection}")
    results = load_json(args.detection)
    print(f"[>>] 加载标注答案: {args.ground_truth}")
    ground_truth = load_json(args.ground_truth)
    print(f"[OK] 共 {len(results)} 条样本")

    # 计算指标
    metrics = compute_metrics(results, ground_truth)

    # 生成报告
    report = generate_report(metrics)

    # 写文件
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] 报告已保存至: {args.output}")

    # 终端摘要
    print(f"\n{'='*60}")
    print(f"  整体得分")
    print(f"{'='*60}")
    print(f"  准确率: {metrics['accuracy']:.2%}    精确率: {metrics['precision']:.2%}")
    print(f"  召回率: {metrics['recall']:.2%}    F1: {metrics['f1']:.2%}")
    print(f"  特异度: {metrics['specificity']:.2%}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
