"""SearchLogQA 数据加载器。"""
from __future__ import annotations

from skillopt.datasets.base import SplitDataLoader


class SearchLogQADataLoader(SplitDataLoader):
    """SearchLogQA 数据加载器。

    每个 split 目录 (train/val/test) 包含一个 items.json，
    格式为从搜索日志提取的 QA 记录列表。
    基类 SplitDataLoader 自动处理文件发现、JSON 解析和 split 分组。
    """

    def load_raw_items(self, data_path: str) -> list[dict]:
        """支持 split_mode='ratio' 模式。"""
        import json

        with open(data_path, encoding="utf-8") as f:
            return json.load(f)
