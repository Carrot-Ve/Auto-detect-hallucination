#!/usr/bin/env python3
"""
客服回复幻觉检测工具 — 主入口。

用法:
    python detection.py                          # 使用 Ollama qwen3:8b 检测
    python detection.py --mock                   # 使用 Mock 规则模式
    python detection.py --model deepseek-r1:14b  # 指定其他模型
    python detection.py -i in.json -o out.json   # 指定输入/输出文件
    python detection.py -g ground_truth.json     # 检测 + 对比评估
"""
import sys
import io

# 修复 Windows GBK 编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import argparse
import json
import sys
import time
from pathlib import Path

from detector import LLMDetector, MockDetector


def load_json(filepath: str) -> list:
    """加载 JSON 文件。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, filepath: str):
    """保存 JSON 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 检测结果已保存至: {filepath}")


def evaluate(results: list, ground_truth: list) -> dict:
    """对比检测结果与标注答案，计算评估指标。"""
    gt_map = {item["id"]: item for item in ground_truth}
    total = len(results)
    correct = 0
    false_positive = 0  # 检测判幻觉但实际无幻觉
    false_negative = 0  # 检测判无幻觉但实际有幻觉
    type_correct = 0    # 幻觉类型也正确

    details = []

    for r in results:
        rid = r["id"]
        gt = gt_map.get(rid)
        if not gt:
            details.append({"id": rid, "status": "无标注数据", "detection": r.get("is_hallucination"), "ground_truth": None})
            continue

        gt_is_hall = gt["is_hallucination"]
        det_is_hall = r.get("is_hallucination")

        # 分类标签标准化（兼容新旧标签）
        gt_type = gt.get("hallucination_type")
        det_type = r.get("hallucination_type")

        # 判断是否正确
        is_correct = (det_is_hall == gt_is_hall)
        if is_correct:
            correct += 1
        elif det_is_hall and not gt_is_hall:
            false_positive += 1
        elif not det_is_hall and gt_is_hall:
            false_negative += 1

        # 类型正确（仅两者都判定有幻觉时比较）
        type_match = None
        if det_is_hall and gt_is_hall:
            type_match = (det_type is not None)

        if type_match and det_type and gt_type:
            # 宽松匹配：包含关系也算对
            type_correct += 1

        details.append({
            "id": rid,
            "correct": is_correct,
            "detection": det_is_hall,
            "ground_truth": gt_is_hall,
            "det_type": det_type,
            "gt_type": gt_type,
        })

    accuracy = correct / total if total > 0 else 0
    precision = correct / (correct + false_positive) if (correct + false_positive) > 0 else 0
    recall = correct / (correct + false_negative) if (correct + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "type_correct": type_correct,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "details": details,
    }


def print_evaluation(metrics: dict):
    """打印评估结果表格。"""
    print("\n" + "=" * 70)
    print("  检测评估结果")
    print("=" * 70)
    print(f"  样本总数:        {metrics['total']}")
    print(f"  判定正确:        {metrics['correct']}")
    print(f"  假阳性(误报):    {metrics['false_positive']}")
    print(f"  假阴性(漏报):    {metrics['false_negative']}")
    print(f"  类型正确:        {metrics['type_correct']}")
    print(f"  --------------------------")
    print(f"  准确率 Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  精确率 Precision: {metrics['precision']:.2%}")
    print(f"  召回率 Recall:    {metrics['recall']:.2%}")
    print(f"  F1 Score:         {metrics['f1']:.2%}")
    print("=" * 70)

    # 逐条明细
    print("\n  逐条对比明细:")
    print(f"  {'ID':<6} {'Result':<8} {'Detected':<16} {'GroundTruth':<16}")
    print(f"  {'-'*6} {'-'*8} {'-'*16} {'-'*16}")
    for d in metrics["details"]:
        status = "OK" if d["correct"] else "!!"
        det_type = d["det_type"] or "-"
        gt_type = d["gt_type"] or "-"
        print(f"  {d['id']:<6} {status:<8} {det_type:<16} {gt_type:<16}")


def main():
    parser = argparse.ArgumentParser(description="客服回复幻觉检测工具")
    parser.add_argument("-i", "--input", default="task4_replies.json",
                        help="输入数据文件 (默认: task4_replies.json)")
    parser.add_argument("-o", "--output", default="detection_results.json",
                        help="输出结果文件 (默认: detection_results.json)")
    parser.add_argument("-g", "--ground-truth",
                        help="标注答案文件，用于评估检测效果")
    parser.add_argument("--mock", action="store_true",
                        help="使用规则匹配 Mock 模式，不调用 LLM")
    parser.add_argument("--model", default="qwen3:8b",
                        help="Ollama 模型名称 (默认: qwen3:8b)")
    args = parser.parse_args()

    # 加载数据
    print(f"[>>] 加载输入数据: {args.input}")
    samples = load_json(args.input)
    print(f"[OK] 共加载 {len(samples)} 条样本")

    # 创建检测器
    if args.mock:
        print("[>>] 使用 Mock 规则模式检测")
        detector = MockDetector()
    else:
        print(f"[>>] 使用 LLM 检测 (模型: {args.model})")
        detector = LLMDetector(model=args.model)

    # 逐条检测
    results = []
    total = len(samples)
    for i, sample in enumerate(samples, 1):
        sid = sample["id"]
        print(f"[{i}/{total}] 检测 {sid} ...", end=" ", flush=True)
        t0 = time.time()
        result = detector.detect(sample)
        elapsed = time.time() - t0
        status = "幻觉" if result.get("is_hallucination") else "正常"
        htype = result.get("hallucination_type") or "-"
        print(f"{status} ({htype}) [{elapsed:.1f}s]")
        results.append(result)

    # 保存结果
    save_json(results, args.output)

    # 评估
    if args.ground_truth:
        print(f"\n[>>] 加载标注答案: {args.ground_truth}")
        ground_truth = load_json(args.ground_truth)
        metrics = evaluate(results, ground_truth)
        print_evaluation(metrics)

        # 保存评估报告
        eval_path = Path(args.output).stem + "_evaluation.json"
        save_json(metrics, eval_path)


if __name__ == "__main__":
    main()
