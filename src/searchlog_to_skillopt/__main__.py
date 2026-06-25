"""CLI 入口：从搜索日志生成 SkillOpt 训练数据。

用法:
    # 仅生成数据
    python -m src.searchlog_to_skillopt --log-dir D:\\docs\\search_logs --output data\\searchlog_qa

    # 生成数据 + 直接训练
    python -m src.searchlog_to_skillopt --log-dir D:\\docs\\search_logs --output data\\searchlog_qa --train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 上
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_dotenv():
    """加载项目根目录的 .env 文件到当前进程环境变量。"""
    import os as _os
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 支持 export KEY=VALUE 和 KEY=VALUE 两种格式
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in _os.environ:
                _os.environ[key] = val


def main():
    parser = argparse.ArgumentParser(
        description="SearchLog → SkillOpt 训练数据生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(r"D:\docs\search_logs"),
        help="搜索日志目录路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/searchlog_qa"),
        help="SkillOpt 数据输出目录",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="训练集比例 (default: 0.6)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例 (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="生成数据后立即启动训练",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="训练轮数 (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=int,
        default=4,
        help="每步最大编辑数 (default: 4)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行 rollout worker 数 (default: 8)",
    )

    args = parser.parse_args()

    # ── Step 1: 提取 ──
    from src.searchlog_to_skillopt.extract import extract_all
    from src.searchlog_to_skillopt.build import build_splits

    print(f"\n{'='*60}")
    print(f"  SearchLog → SkillOpt 数据生成")
    print(f"{'='*60}")
    print(f"  日志目录:  {args.log_dir}")
    print(f"  输出目录:  {args.output}")

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"\n  ❌ 错误: 日志目录不存在: {log_dir}")
        sys.exit(1)

    print(f"\n  [1/2] 解析搜索日志...")
    records = extract_all(log_dir)
    print(f"        提取到 {len(records)} 条有效 QA 记录")

    if not records:
        print(f"\n  ❌ 未提取到任何有效记录，请检查日志目录")
        sys.exit(1)

    # 统计
    sources = {}
    domains = {}
    for r in records:
        s = r.get("source", "?")
        sources[s] = sources.get(s, 0) + 1
        d = r.get("domain", "?")
        domains[d] = domains.get(d, 0) + 1
    print(f"        来源分布: {sources}")
    print(f"        领域分布: {domains}")

    print(f"\n  [2/2] 生成训练/验证/测试集...")
    counts = build_splits(
        records,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"        train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"        → 数据已写入 {args.output.resolve()}")

    # ── Step 2: 可选训练 ──
    if args.train:
        print(f"\n{'='*60}")
        print(f"  启动 SkillOpt 训练")
        print(f"{'='*60}")

        # 确保 .env 已加载到当前进程环境
        _load_dotenv()

        config_path = _PROJECT_ROOT / "configs" / "searchlog_qa" / "default.yaml"
        if not config_path.exists():
            print(f"\n  ❌ 配置文件不存在: {config_path}")
            sys.exit(1)

        # 直接调用训练入口
        from skillopt.config import load_config, flatten_config, is_structured

        cfg = load_config(str(config_path))
        flat = flatten_config(cfg) if is_structured(cfg) else cfg

        # 从 .env 自动补全 DeepSeek 配置
        import os as _os
        if not flat.get("azure_openai_endpoint"):
            flat["azure_openai_endpoint"] = _os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not flat.get("azure_openai_api_key"):
            flat["azure_openai_api_key"] = _os.environ.get("AZURE_OPENAI_API_KEY", "")
        auth_mode = _os.environ.get("AZURE_OPENAI_AUTH_MODE", "")
        if auth_mode and "azure_openai_auth_mode" not in flat:
            flat["azure_openai_auth_mode"] = auth_mode

        # 覆盖 CLI 指定的参数
        if args.epochs:
            flat["num_epochs"] = args.epochs
        if args.learning_rate:
            flat["edit_budget"] = args.learning_rate
        flat.setdefault("optimizer_backend", "openai_chat")
        flat.setdefault("target_backend", "openai_chat")
        flat.setdefault("out_root", str(_PROJECT_ROOT / "outputs" / "searchlog_qa"))

        # 导入并运行训练
        from skillopt.model import (
            configure_azure_openai,
            set_optimizer_backend,
            set_target_backend,
            set_optimizer_deployment,
            set_target_deployment,
        )

        # 配置 DeepSeek
        configure_azure_openai(
            endpoint=flat.get("azure_openai_endpoint"),
            api_key=flat.get("azure_openai_api_key"),
            auth_mode=flat.get("azure_openai_auth_mode", "openai_compatible"),
        )
        set_optimizer_backend("openai_chat")
        set_target_backend("openai_chat")
        set_optimizer_deployment(flat.get("optimizer_model", "deepseek-chat"))
        set_target_deployment(flat.get("target_model", "deepseek-chat"))

        # 构建 adapter 和 trainer
        from scripts.train import get_adapter
        from skillopt.engine.trainer import ReflACTTrainer

        adapter = get_adapter(flat)
        trainer = ReflACTTrainer(flat, adapter)
        summary = trainer.train()

        print(f"\n{'='*60}")
        print(f"  训练完成!")
        print(f"  输出目录: {flat['out_root']}")
        if summary.get("test_hard") is not None:
            print(f"  最终测试准确率: {summary['test_hard']:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
