"""
LLM 检测器 — 通过 Ollama API 调用 qwen3:8b 进行幻觉检测。
"""

import json
import re
from .prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class LLMDetector:
    """使用 Ollama 本地模型进行幻觉检测。"""

    def __init__(self, model: str = "qwen3:8b", host: str = "http://127.0.0.1:11434", timeout: int = 120):
        if not HAS_OLLAMA:
            raise ImportError("请安装 ollama 库: pip install ollama")
        self.model = model
        self.host = host
        self.timeout = timeout
        self._client = ollama.Client(host=host)

    def detect(self, sample: dict) -> dict:
        """对单条样本进行幻觉检测。

        Args:
            sample: {"id": str, "user_question": str, "system_reply": str, "knowledge_base": str}

        Returns:
            检测结果 dict，与 ground_truth 格式对齐
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            user_question=sample["user_question"],
            system_reply=sample["system_reply"],
            knowledge_base=sample["knowledge_base"],
        )

        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.1, "num_predict": 2048},
            )

            raw_output = response["message"]["content"].strip()
            result = self._parse_output(raw_output, sample["id"])

        except Exception as e:
            result = {
                "id": sample["id"],
                "is_hallucination": None,
                "hallucination_type": None,
                "secondary_types": [],
                "severity": None,
                "deviation_mode": None,
                "detail": f"检测失败: {str(e)}",
                "error": str(e),
            }

        return result

    def _parse_output(self, raw: str, sample_id: str) -> dict:
        """从 LLM 原始输出中提取 JSON 结果。"""
        # 1. 剥离 <think>...</think> 推理块
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        cleaned = cleaned.strip() or raw  # 如果删除后为空，回退原文

        # 2. 提取 JSON（支持嵌套 {}，如 secondary_types: [...])
        json_str = None
        # 找最外层 JSON 对象
        brace_start = cleaned.find('{')
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = cleaned[brace_start:i + 1]
                        break

        if json_str is None:
            return self._fallback_parse(raw, sample_id)

        try:
            parsed = json.loads(json_str)
            return {
                "id": sample_id,
                "is_hallucination": parsed.get("is_hallucination"),
                "hallucination_type": parsed.get("hallucination_type"),
                "secondary_types": parsed.get("secondary_types", []),
                "severity": parsed.get("severity"),
                "deviation_mode": parsed.get("deviation_mode"),
                "detail": parsed.get("detail", ""),
            }
        except json.JSONDecodeError:
            return self._fallback_parse(raw, sample_id)

    def _fallback_parse(self, raw: str, sample_id: str) -> dict:
        """JSON 解析失败时的兜底处理。"""
        # 剥离 think 块后再匹配
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)

        # 从清理后的文本匹配 is_hallucination
        hall_match = re.search(r'"is_hallucination"\s*:\s*(true|false)', cleaned)
        is_hall = hall_match.group(1) == "true" if hall_match else None

        result = {
            "id": sample_id,
            "is_hallucination": is_hall,
            "hallucination_type": None,
            "secondary_types": [],
            "severity": None,
            "deviation_mode": None,
            "detail": "JSON解析失败，原始输出见raw_output",
            "raw_output": raw,
        }

        if is_hall is None:
            # 如果连 boolean 都找不到，从推理文本判断
            # 若 think 中提到矛盾/编造/不存在等词，倾向判定为幻觉
            think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
            think_text = think_match.group(1) if think_match else ""
            fabrication_signals = ["编造", "虚构", "矛盾", "不一致", "不存在", "未接入", "杜撰"]
            result["is_hallucination"] = any(s in think_text for s in fabrication_signals)

        # 从清理后文本匹配分类关键词
        type_keywords = [
            "安全合规幻觉", "系统能力幻觉", "产品参数幻觉",
            "政策规则幻觉", "优惠活动幻觉", "实体信息幻觉", "关键信息遗漏",
        ]
        for kw in type_keywords:
            if kw in cleaned:
                result["hallucination_type"] = kw
                break

        severity_map = [
            ("致命", "致命"),
            ("严重", "严重"),
            ("高", "高"),
            ("中", "中"),
            ("低", "低"),
        ]
        for kw, sv in severity_map:
            if kw in cleaned:
                result["severity"] = sv
                break

        return result
