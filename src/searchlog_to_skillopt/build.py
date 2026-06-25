"""将提取的 QA 记录划分 train/val/test 并写入 SkillOpt 数据格式。"""
from __future__ import annotations

import json
import random
from pathlib import Path


def build_splits(
    records: list[dict],
    output_dir: str | Path,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    use_all: bool = True,
) -> dict[str, int]:
    """将 records 划分为 train/val/test 并写入 items.json。

    Parameters
    ----------
    records : list[dict]
        提取的 QA 记录。
    output_dir : str | Path
        输出根目录（会创建 train/, val/, test/ 子目录）。
    train_ratio, val_ratio : float
        train 和 val 的比例，剩余为 test。
    seed : int
        随机种子。
    use_all : bool
        是否使用全部记录（包括 CLI 模式），默认 True 会做分层采样。

    Returns
    -------
    dict[str, int]
        {"train": N, "val": N, "test": N}
    """
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    # 分层：按 domain 分组，确保每个 split 都包含各 domain 的样本
    domains: dict[str, list[dict]] = {}
    for r in records:
        domain = r.get("domain", "unknown")
        domains.setdefault(domain, []).append(r)

    train_items, val_items, test_items = [], [], []

    for domain, items in domains.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # 确保至少每个 split 有 1 条（如果总数允许）
        if n < 3:
            # 太少就全放 train
            train_items.extend(items)
        else:
            train_items.extend(items[:n_train])
            val_items.extend(items[n_train:n_train + n_val])
            test_items.extend(items[n_train + n_val:])

    # 写入
    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "items.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    counts = {"train": len(train_items), "val": len(val_items), "test": len(test_items)}

    # 写 split_manifest
    manifest = {
        "source": "search_logs",
        "split_ratio": f"{train_ratio}:{val_ratio}:{1-train_ratio-val_ratio:.1f}",
        "seed": seed,
        "counts": counts,
        "item_fields": [
            "id", "question", "context", "gold_answer",
            "source", "search_mode", "domain",
        ],
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return counts
