"""解析搜索日志 .md 文件，提取结构化 QA 数据。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """解析 YAML frontmatter（--- 之间的部分）。"""
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    front = m.group(1)
    data: dict[str, Any] = {}
    for line in front.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # 转换常见类型
        if val.lower() in ("true", "yes"):
            val = True
        elif val.lower() in ("false", "no"):
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        data[key] = val
    return data


def _extract_section(text: str, heading: str) -> str:
    """提取 Markdown 中指定标题下的内容。"""
    # 匹配 "# Heading\n...\n# " 或 "# Heading\n...(到文件尾)"
    pattern = rf"^# {re.escape(heading)}\s*\n(.*?)(?=^# |\Z)"
    m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    if not m:
        return ""
    return m.group(1).strip()


def _is_meaningful_response(response: str) -> bool:
    """判断回答是否包含实质内容（而非单纯的文档列表）。"""
    if not response or len(response.strip()) < 30:
        return False
    # CLI 模式的 response 通常是"找到 N 个结果: doc1, doc2..."
    if re.match(r"^找到\s*\d+\s*个结果", response.strip()):
        return False
    # 排除没有找到信息的回答
    no_info_patterns = [
        r"没有找到.*相关",
        r"未找到.*信息",
        r"无法.*回答",
        r"not found",
    ]
    for p in no_info_patterns:
        if re.search(p, response, re.IGNORECASE):
            return False
    return True


def extract_from_md(md_path: str | Path) -> dict | None:
    """从单个 .md 日志文件提取 QA 对。

    Returns
    -------
    dict | None
        包含 id, question, context, answer, ... 的字典；无效时返回 None。
    """
    path = Path(md_path)
    text = path.read_text(encoding="utf-8")

    meta = _parse_frontmatter(text)
    if not meta:
        return None

    instruction = _extract_section(text, "Instruction")
    response = _extract_section(text, "Response")
    reasoning = _extract_section(text, "Reasoning Trace")
    retrieved = _extract_section(text, "Retrieved Context")

    # 上下文：优先使用 Retrieved Context，其次 Reasoning Trace
    context = retrieved if retrieved.strip() else reasoning

    if not instruction.strip():
        return None

    # 过滤：只用 agent 模式且有实质回答的日志做训练数据
    source = meta.get("source", "")
    if source == "agent" and not _is_meaningful_response(response):
        return None

    session_id = meta.get("session_id", path.stem)
    search_mode = meta.get("search_mode", "unknown")
    index_path = meta.get("index_path", "")
    success = meta.get("success", True)

    # 提取简短域名（用于 task_type 分层）
    if index_path:
        # 标准化路径：反斜杠 → 正斜杠，处理双反斜杠
        normalized = index_path.replace("\\\\", "/").replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        # 找 "raw" 后的第一级目录名
        try:
            raw_idx = parts.index("raw")
            domain = parts[raw_idx + 1] if raw_idx + 1 < len(parts) else "unknown"
        except ValueError:
            domain = parts[-2] if len(parts) >= 2 else parts[-1] if parts else "unknown"
    else:
        domain = "unknown"

    return {
        "id": session_id,
        "question": instruction.strip(),
        "context": context.strip(),
        "gold_answer": response.strip(),
        "source": source,
        "search_mode": search_mode,
        "domain": domain,
        "index_path": index_path,
        "success": bool(success),
        "timestamp": meta.get("timestamp", ""),
        "model": meta.get("model", ""),
    }


def extract_all(log_dir: str | Path) -> list[dict]:
    """从日志目录提取所有有效 QA 对。

    Parameters
    ----------
    log_dir : str | Path
        搜索日志目录路径。

    Returns
    -------
    list[dict]
        按时间排序的有效 QA 记录列表。
    """
    log_dir = Path(log_dir)
    records: list[dict] = []

    for md_file in sorted(log_dir.glob("srch_*.md")):
        record = extract_from_md(md_file)
        if record:
            records.append(record)

    # 按时间排序
    records.sort(key=lambda r: r.get("timestamp", ""))
    return records
