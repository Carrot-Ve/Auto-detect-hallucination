"""
Mock 检测器 — 基于规则的幻觉检测，不调用 LLM API。
用于快速验证流程、低成本粗略筛选。
"""

import re


# ── 能力越界特征词 ──
CAPABILITY_PATTERNS = [
    r"已(帮|为)您(查|修改|升级|处理|操作|设置|添加|删除|取消|提交|申请|发放)",
    r"(已经|已)(查|修改|升级|处理|操作|设置)了",
    r"(帮|为)您(查|修改|升级|处理|操作|设置|发放)",
    r"(预计|大约|大概).{0,10}(到账|送达|联系|到达)",
    r"已(升级|提交|转交)(为|到|至|给)",
]

# ── 安全误导特征词 ──
SAFETY_PATTERNS = [
    r"(孕妇|怀孕|哺乳|备孕).{0,20}(可以|放心|安全|没问题|能用|可用|适合)",
    r"(婴儿|宝宝|新生儿|小孩|儿童).{0,20}(可以|放心|安全|没问题|能用|可用)",
    r"可以(放心|大胆).*(用|吃|服用|使用)",
    r"(绝对|百分百|确定)(安全|无害|没问题|不过敏)",
    r"无(任何|副|毒).*(作用|反应)",
]

# ── 实体信息编造特征 ──
ENTITY_PATTERNS = [
    r'(地址|电话|手机号|联系人|收件人|经理|主管|门店|线下|实体店|专柜)',
]

# ── 数字/参数提取 ──
NUMBER_PATTERN = re.compile(r'(\d+\.?\d*)\s*(天|小时|年|月|元|块|折|%|mm|cm|克|斤|个|次|代|\.\d+)')
VERSION_PATTERN = re.compile(r'(\d+\.\d+)\s*(版本|蓝牙|接口|系统)')


class MockDetector:
    """基于规则的粗略幻觉检测，不调用 LLM。"""

    def detect(self, sample: dict) -> dict:
        sid = sample["id"]
        reply = sample.get("system_reply", "")
        kb = sample.get("knowledge_base", "")

        # 1. 检测能力越界
        cap_match = self._check_capability(reply, kb)
        if cap_match:
            return self._result(sid, True, "系统能力幻觉", "严重", "能力虚构", cap_match)

        # 2. 检测安全误导
        safety_match = self._check_safety(reply, kb)
        if safety_match:
            return self._result(sid, True, "安全合规幻觉", "致命", "完全编造", safety_match)

        # 3. 检测数字/参数矛盾
        param_match = self._check_param_contradiction(reply, kb)
        if param_match:
            return self._result(sid, True, "产品参数幻觉", "高", "部分歪曲", param_match)

        # 4. 检测实体信息编造（简单关键词）
        entity_match = self._check_entity_fabrication(reply, kb)
        if entity_match:
            return self._result(sid, True, "实体信息幻觉", "中", "完全编造", entity_match)

        # 5. 检测政策/优惠矛盾（关键词）
        policy_match = self._check_policy_contradiction(reply, kb)
        if policy_match:
            return self._result(sid, True, "政策规则幻觉", "高", "部分歪曲", policy_match)

        # 默认：无法通过规则判定 → 标记为无幻觉（规则能力有限）
        return self._result(sid, False, None, None, None, "Mock规则未检测到明显幻觉特征，可能遗漏")

    def _check_capability(self, reply: str, kb: str) -> str | None:
        """检查能力越界：KB中标注'未接入'/'不具备'，但回复声称执行了操作。"""
        kb_has_no_capability = any(kw in kb for kw in ["未接入", "不具备", "需人工", "不可", "无（"])
        if not kb_has_no_capability:
            return None

        for pattern in CAPABILITY_PATTERNS:
            if re.search(pattern, reply):
                return f"回复使用了能力越界表述'{re.search(pattern, reply).group(0)}'，但知识库表明系统不具备该能力"
        return None

    def _check_safety(self, reply: str, kb: str) -> str | None:
        """检查安全误导：KB中有安全警告但回复说安全。"""
        kb_has_warning = any(kw in kb for kw in ["孕妇", "建议咨询医生", "谨慎", "不宜", "禁用", "慎用", "过敏"])
        if not kb_has_warning:
            return None

        for pattern in SAFETY_PATTERNS:
            if re.search(pattern, reply):
                return f"回复声称安全可用，但知识库包含安全警告"
        return None

    def _check_param_contradiction(self, reply: str, kb: str) -> str | None:
        """检查参数矛盾：数字/参数在 reply 和 KB 中不一致。"""
        reply_nums = NUMBER_PATTERN.findall(reply)
        kb_nums = NUMBER_PATTERN.findall(kb)

        # 提取含单位的数字对
        reply_set = {f"{n[0]}{n[1]}" for n in reply_nums}
        kb_set = {f"{n[0]}{n[1]}" for n in kb_nums}

        # 找出 reply 中有但 KB 中无的数字+单位组合
        # 并且 KB 中存在同单位但不同数值的项
        for r_num, r_unit in reply_nums:
            for k_num, k_unit in kb_nums:
                if r_unit == k_unit and r_num != k_num:
                    # 排除"无"关键词的情况（KB中"无"不是数值）
                    return f"参数矛盾：回复声称{r_num}{r_unit}，知识库记录{k_num}{k_unit}"
        return None

    def _check_entity_fabrication(self, reply: str, kb: str) -> str | None:
        """检查实体信息编造。"""
        # KB明确说"无"但reply声称"有"
        kb_explicit_none = re.findall(r'(无|没有|纯线上|不存在)\s*(线下|门店|实体|关联|该)', kb)
        if kb_explicit_none:
            for pattern in ENTITY_PATTERNS:
                m = re.search(pattern, reply)
                if m:
                    return f"回复声称了{m.group(1)}信息，但知识库表明该实体不存在"
        return None

    def _check_policy_contradiction(self, reply: str, kb: str) -> str | None:
        """检查政策/优惠矛盾。"""
        # 优惠活动检测
        reply_has_coupon = any(kw in reply for kw in ["满减", "优惠券", "折扣", "学生", "会员", "专属价"])
        kb_no_coupon = any(kw in kb for kw in ["无", "当前无", "暂无"])
        if reply_has_coupon and kb_no_coupon:
            return "回复声称存在优惠活动，但知识库表明无此活动或已明确限定范围"

        # 支付/发票等政策关键词矛盾
        policy_keywords = {
            "纸质发票": "暂不支持纸质发票",
            "货到付款": "不支持货到付款",
            "顺丰": "中通/韵达/圆通",
        }
        for r_kw, kb_kw in policy_keywords.items():
            if r_kw in reply and kb_kw in kb:
                return f"回复声称'{r_kw}'，但知识库写明'{kb_kw}'"
        return None

    def _result(self, sid, is_hall, h_type, severity, dev_mode, detail) -> dict:
        d = {
            "id": sid,
            "is_hallucination": is_hall,
            "hallucination_type": h_type,
            "secondary_types": [],
            "severity": severity,
            "deviation_mode": dev_mode,
            "detail": detail,
            "detector": "mock",
        }
        return d
