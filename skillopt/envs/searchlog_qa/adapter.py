"""SearchLogQA 环境适配器。"""
from __future__ import annotations

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.searchlog_qa.dataloader import SearchLogQADataLoader
from skillopt.envs.searchlog_qa.rollout import run_batch


class SearchLogQAAdapter(EnvAdapter):
    """企业文档搜索 QA 环境适配器。

    将搜索日志中的 QA 对作为训练数据，
    训练 Agent 如何更好地基于搜索上下文回答问题。
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "6:2:2",
        split_seed: int = 42,
        split_output_dir: str = "",
        workers: int = 8,
        analyst_workers: int = 8,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 16384,
        exec_timeout: int = 120,
    ) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)
        self.exec_timeout = int(exec_timeout)
        self.dataloader = SearchLogQADataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        return run_batch(
            items=env_manager,
            out_root=out_dir,
            skill_content=skill_content,
            workers=self.workers,
            max_completion_tokens=self.max_completion_tokens,
            exec_timeout=self.exec_timeout,
            **kwargs,
        )

    def get_task_types(self) -> list[str]:
        # 从数据中收集所有 domain
        seen: list[str] = []
        for items in [self.dataloader.train_items, self.dataloader.val_items, self.dataloader.test_items]:
            for item in items:
                d = str(item.get("domain", "searchlog_qa"))
                if d not in seen:
                    seen.append(d)
        return seen or ["searchlog_qa"]
