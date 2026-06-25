"""SearchLogQA 评分器 — 基于 token overlap 的中文答案评估。"""
from __future__ import annotations

import re
import string
from collections import Counter


def _normalize(text: str) -> str:
    """中文友好的文本规范化。"""
    text = text.lower().strip()
    # 移除标点（保留中文）
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    # 移除英文停用词
    for w in ("a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "for"):
        text = re.sub(rf"\b{w}\b", " ", text)
    return " ".join(text.split())


def extract_answer(text: str) -> str:
    """从模型输出中提取答案。

    支持 <answer>...</answer> 标签，否则返回全文。
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 尝试最后一段非空内容
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    # 跳过以 "找到 X 个结果" 开头的行
    meaningful = [ln for ln in lines if not re.match(r"^找到\s*\d+\s*个结果", ln)]
    if meaningful:
        return meaningful[-1]
    return text.strip()


def _token_f1(pred: str, gold: str) -> float:
    """基于 token 的 F1 分数。"""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _is_empty_answer(text: str) -> bool:
    """判断是否为空/无效回答。"""
    text = text.strip()
    if not text or len(text) < 10:
        return True
    no_info = [
        "没有找到", "未找到", "无法回答", "无法提供", "暂无相关",
        "not found", "no information", "no relevant",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in no_info)


def evaluate(prediction_text: str, item: dict) -> dict:
    """评估单条预测。

    Parameters
    ----------
    prediction_text : str
        模型原始输出。
    item : dict
        数据项，需包含 gold_answer 字段。

    Returns
    -------
    dict with: hard (0/1), soft (float), predicted_answer, gold_answer
    """
    predicted = extract_answer(prediction_text)
    gold_answer = item.get("gold_answer", "")

    # 空答案 → 0 分
    if _is_empty_answer(predicted):
        return {
            "hard": 0,
            "soft": 0.0,
            "predicted_answer": predicted,
            "gold_answer": gold_answer,
        }

    # 计算 token F1 作为 soft 分数
    soft = _token_f1(predicted, gold_answer)

    # hard: soft > 0.3 且不是空答案即算对
    hard = 1 if soft >= 0.3 else 0

    return {
        "hard": hard,
        "soft": round(soft, 4),
        "predicted_answer": predicted,
        "gold_answer": gold_answer,
    }
