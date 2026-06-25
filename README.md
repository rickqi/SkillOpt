# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

*Train agent skills like you train neural networks — with epochs, (mini-)batchsize, learning rates, and validation gates — but without touching model weights.*

[![Project Page](https://img.shields.io/badge/Project%20Page-SkillOpt-8dbb3c)](https://microsoft.github.io/SkillOpt/) [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2605.23904) [![Project Video](https://img.shields.io/badge/Project%20Video-Watch%20Demo-ff0000)](https://youtu.be/JUBMDTCiM0M) [![PyPI](https://img.shields.io/badge/PyPI-skillopt-green.svg)](https://pypi.org/project/skillopt/) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <a href="https://trendshift.io/repositories/38498?utm_source=trendshift-badge&utm_medium=badge&utm_campaign=badge-trendshift-38498" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/38498/daily?language=Python" alt="microsoft%2FSkillOpt | Trendshift" width="250" height="55"/></a>
  <a href="https://trendshift.io/repositories/38498?utm_source=trendshift-badge&utm_medium=badge&utm_campaign=badge-trendshift-38498" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/38498/weekly?language=Python" alt="microsoft%2FSkillOpt | Trendshift" width="250" height="55"/></a>
</p>

> 📖 **For installation, data preparation, training/eval commands, the full configuration reference, and framework internals, see the [Documentation & Reproduction Guide](https://microsoft.github.io/SkillOpt/docs/guideline.html)** (rendered on GitHub Pages).

---

## 中文概述 / Chinese Overview

### SkillOpt 是什么？

**SkillOpt** 是一个将深度学习训练范式引入 Agent 提示词优化的框架。它的核心思想是：**将"技能文档"（一份 Markdown 格式的自然语言指令）视为冻结模型的可训练参数**，用 epochs、learning rate、batch size、validation gate 等概念来迭代优化它——整个过程**不修改模型权重**。

传统做法是靠人工编写 Prompt、或用强模型一次性生成、或通过松散的自我修订来"进化"——这些方法都不像深度学习优化器那样稳定可靠。SkillOpt 改变了这一点。

### 核心类比：深度学习 ↔ 技能优化

| 深度学习概念 | SkillOpt 对应 |
|---|---|
| 模型权重（weights） | 技能文档（Markdown，约 300-2,000 tokens） |
| 前向传播（forward pass） | **Rollout**：目标 Agent 使用当前技能执行任务 |
| 损失/梯度（loss/gradient） | **Reflect**：优化器模型分析执行轨迹，生成编辑补丁 |
| 梯度裁剪（gradient clipping） | **Select**：按编辑预算（learning_rate）裁剪编辑数量 |
| SGD 步进（SGD step） | **Update**：将编辑应用到技能文档 |
| 验证集（validation set） | **Gate**：在 held-out 集上评估，只有严格提升才接受 |
| 学习率调度（LR schedule） | `lr_scheduler`：cosine / linear / constant / autonomous |
| Epoch 训练 | 多 epoch 配合 Slow Update（类 EMA）和 Meta Skill（跨 epoch 记忆） |

### 训练管线：ReflACT 六阶段

```
  [① Rollout]    [② Reflect]   [③ Aggregate]   [④ Select]   [⑤ Update]   [⑥ Gate]
  目标Agent       优化器分析       分层合并        排序裁剪      应用编辑      验证门控
  执行任务 →      生成补丁 →      多个补丁 →      Top-K →      改技能文档 →  合格才接受
```

| 阶段 | 职责 | 类比 |
|---|---|---|
| **① Rollout** | 目标 Agent 用当前技能执行一批任务，收集得分和完整轨迹 | 前向传播 |
| **② Reflect** | 优化器模型以 minibatch 方式分析失败/成功轨迹，输出结构化编辑补丁 | 梯度计算 |
| **③ Aggregate** | 将多个独立补丁通过分层 LLM 合并为一个统一补丁 | 梯度聚合 |
| **④ Select** | 按重要性排序，根据当前 editing budget 裁剪（类比梯度裁剪） | 梯度裁剪 |
| **⑤ Update** | 将选中的编辑（append/insert/replace/delete）应用到技能文档 | 参数更新 |
| **⑥ Gate** | 在 held-out 验证集上评估候选技能，只有**严格提升**才接受；否则回退 | 验证集判断 |

**Epoch 边界**还包含两个机制：
- **Slow Update**：对前后 epoch 的技能进行纵向对比，生成高层指导注入技能文档（类比 EMA / 正则化）
- **Meta Skill**：跨 epoch 积累策略记忆，为后续反思提供上下文

### 项目模块结构

| 目录 | 职能 | 关键文件 |
|---|---|---|
| `skillopt/engine/` | **训练主循环** | `trainer.py`（2380 行），调度全部 6 阶段 |
| `skillopt/model/` | **多后端 API 层** | 路由到 OpenAI/Azure、Claude、Qwen、MiniMax；+ Codex/Claude Code CLI 执行驱动 |
| `skillopt/gradient/` | **梯度计算** | `reflect.py`（minibatch 轨迹分析）、`aggregate.py`（分层补丁合并）|
| `skillopt/optimizer/` | **优化器** | 编辑应用、排名裁剪、学习率调度、Slow Update、Meta Skill |
| `skillopt/evaluation/` | **验证门控** | 接受/拒绝判定（hard/soft/mixed 三种指标）|
| `skillopt/envs/` | **11+ Benchmark 适配器** | 每个 benchmark 提供 dataloader、rollout、reflection、评分 |
| `skillopt/datasets/` | **数据加载** | 训练/选择/测试集划分 |
| `skillopt/prompts/` | **20+ 提示词模板** | analyst_error/success、merge、ranking、meta_skill 等 |
| `configs/` | **YAML 配置** | 支持 `_base_` 继承、结构化/扁平两种格式 |
| `skillopt_sleep/` | **部署时伴侣（Preview）** | 零依赖独立包，为本地 Agent 提供夜间自我进化 |
| `skillopt_webui/` | **WebUI 监控面板** | Gradio 应用，可视化配置和训练 |

### 关键技术创新

1. **文本学习率（Textual Learning Rate）**：用自然语言编辑预算（editing budget）替代数值学习率，支持 cosine/linear/constant/autonomous 调度策略
2. **Minibatch 反思**：轨迹分组批量分析（而非逐条），类 mini-batch SGD——比逐样本分析更稳定
3. **严格验证门控**：只在 held-out 集上**严格提升**时才接受修改，杜绝退化——类比 early stopping
4. **零推理开销部署**：最终产物只是一个紧凑的 `.md` 文件，部署时不需要任何额外的模型调用
5. **Skill 跨模型迁移**：优化后的 Skills 可跨模型大小、跨执行框架、跨相邻 Benchmark 直接使用
6. **Skill-Aware Reflection (EmbodiSkill)**：分离技能文档缺陷 vs. 执行失误，通过受保护附录区实现
7. **执行驱动（Exec Harness）**：支持直接在 Codex CLI / Claude Code CLI 的真实 Agent 编码循环中训练

### 评估结果（论文数据）

- **52 个评估单元**（6 Benchmark × 7 模型 × 3 执行驱动）全部达到**最佳或并列最佳**
- 在 GPT-5.5 上，相比无 Skill 的基线：
  - 直接对话：**+23.5** 个百分点
  - Codex Agent 循环内：**+24.8** 个百分点
  - Claude Code 内：**+19.1** 个百分点
- 优化后的 Skill 产物可跨模型大小、跨执行框架、跨相邻基准**直接迁移**而无需再优化

### SkillOpt-Sleep 😴（部署时夜间自我进化）

SkillOpt-Sleep 将 SkillOpt 的训练纪律应用到你的**日常使用**中。它给你的本地编码 Agent 一个"夜间睡眠周期"：

```
收割对话记录 → 挖掘重复任务 → 离线重放 → 门控验证 → 暂存更新技能 → 用户审查后应用
```

支持 **Claude Code**、**Codex**、**Copilot** 三个平台，零依赖独立运行。详见 [`docs/sleep/README.md`](docs/sleep/README.md)。

---

## 快速开始 / Quick Start

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt

# 安装（推荐开发模式）
pip install -e .

# 或通过 PyPI 安装
pip install skillopt
```

**Python ≥ 3.10** 必须。可选依赖见下表：

| 用途 | 安装命令 |
|---|---|
| ALFWorld Benchmark | `pip install -e ".[alfworld]"` |
| Claude 后端 | `pip install -e ".[claude]"` |
| Qwen 本地模型 | `pip install -e ".[qwen]"` |
| WebUI 面板 | `pip install -e ".[webui]"` |
| 开发工具 | `pip install -e ".[dev]"` |

### 2. 配置 API 凭证

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

支持的模型后端：

| 后端 | 环境变量 | 说明 |
|---|---|---|
| **Azure OpenAI**（默认） | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` | 支持 API Key / Azure CLI / Managed Identity 三种认证 |
| **OpenAI 直连** | `AZURE_OPENAI_ENDPOINT=https://api.openai.com/v1` + `AZURE_OPENAI_AUTH_MODE=openai_compatible` | 兼容 OpenAI 原生 API |
| **Anthropic Claude** | `ANTHROPIC_API_KEY` | 需安装 `[claude]` 可选依赖 |
| **Qwen 本地模型** | `QWEN_CHAT_BASE_URL` | 通过 vLLM 部署 |
| **MiniMax** | `MINIMAX_BASE_URL` + `MINIMAX_API_KEY` | 国产模型支持 |

### 3. 运行第一个实验

以 **SearchQA**（最快完成，约 30 分钟）为例：

```bash
# 查看配置
cat configs/searchqa/default.yaml

# 开始训练
python scripts/train.py --config configs/searchqa/default.yaml
```

训练过程中的典型输出：

```
[Step 1/8] Rollout: 20 items, 4 workers...
[Step 1/8] Score: 0.65 → Reflect...
[Step 1/8] 6 edit patches generated
[Step 1/8] Selected 4 edits (lr=8, cosine → 7.7)
[Step 1/8] Gate: val score 0.68 > 0.65 ✓ ACCEPT
```

### 4. 评估最优技能

```bash
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa/<run_id>/skills/best_skill.md
```

### 5. 使用 WebUI（可选）

```bash
pip install -e ".[webui]"
python -m skillopt_webui.app
# 打开 http://localhost:7860
```

### 6. 关键参数速查

| 参数 | 默认值 | 含义 | 深度学习类比 |
|---|---|---|---|
| `train.num_epochs` | 4 | 训练轮数 | epochs |
| `train.batch_size` | 40 | 每步任务数 | batch size |
| `gradient.minibatch_size` | 8 | 反思分组大小 | mini-batch |
| `optimizer.learning_rate` | 4 | 每步最大编辑数 | learning rate |
| `optimizer.lr_scheduler` | cosine | 编辑预算调度策略 | LR schedule |
| `optimizer.use_slow_update` | true | epoch 边界纵向对比更新 | momentum/EMA |
| `optimizer.use_meta_skill` | true | 跨 epoch 策略记忆 | optimizer state |
| `evaluation.use_gate` | true | 验证门控开关 | validation |

CLI 覆写示例：

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  optimizer.learning_rate=16 \
  optimizer.lr_scheduler=linear \
  gradient.analyst_workers=8
```

### 7. 内置 Benchmark 一览

| Benchmark | 类型 | 难度 | 典型耗时 |
|---|---|---|---|
| **SearchQA** | 开放域问答 | ⭐ 简单 | ~30 分钟 |
| **DocVQA** | 文档问答 | ⭐⭐ 中等 | ~2 小时 |
| **ALFWorld** | 具身 AI | ⭐⭐⭐ 困难 | ~3 小时 |
| **OfficeQA** | 企业问答 | ⭐⭐ 中等 | ~2 小时 |
| **LiveMathematicianBench** | 数学推理 | ⭐⭐⭐ 困难 | ~3 小时 |
| **SpreadsheetBench** | 电子表格操作 | ⭐⭐ 中等 | ~2 小时 |
| **SWEBench** | 软件工程 | ⭐⭐⭐ 困难 | 数小时 |
| + BabyVision, MMRB, MathVerse, SealQA 等 |

### 8. 训练产物结构

```
outputs/<benchmark>/<run_id>/
├── steps/
│   ├── step_0001/
│   │   ├── candidate_skill.md    # 本步候选技能
│   │   ├── step_record.json      # 步记录
│   │   └── trajectory_digest.json # 轨迹摘要
│   └── step_0002/
├── slow_update/                   # Slow Update 产物
├── meta_skill/                    # Meta Skill 产物
├── skills/                        # 各步技能快照
├── best_skill.md                  # ★ 最终产物：最佳技能文档
├── history.json                   # 训练历史记录
└── config.yaml                    # 回显配置
```

**最终产物 `best_skill.md` 可以直接作为 System Prompt 部署使用——零额外推理开销。**

---

## News 🔥🔥🔥
- **[2026-06-15]** 😴 **SkillOpt-Sleep (preview)** — a nightly offline self-evolution companion for local coding agents (Claude Code / Codex / Copilot): review past sessions, replay recurring tasks, and consolidate validated skills behind a held-out gate. See **[`docs/sleep/README.md`](docs/sleep/README.md)** for what it is, how to use it, and results.
- **[2026-06-03]** 🎉 **[gbrain](https://github.com/garrytan/gbrain), [gbrain-evals](https://github.com/garrytan/gbrain-evals/blob/main/docs/benchmarks/2026-06-03-skillopt.md), and [darwin-skill](https://github.com/alchaincyf/darwin-skill) have all integrated SkillOpt.**
- **[2026-06-02]** 🎉 **SkillOpt [v0.1.0](https://github.com/microsoft/SkillOpt/releases/tag/v0.1.0) is now available on [PyPI](https://pypi.org/project/skillopt/)!** Install with `pip install skillopt`. This initial release includes the full training loop (rollout → reflect → aggregate → select → update → evaluate), multi-backend support (OpenAI / Azure / Claude / Qwen / MiniMax), six built-in benchmarks, and WebUI dashboard.

---

## Overview (English)

Modern agent skills are usually hand-crafted, generated one-shot by a strong
LLM, or evolved through loosely controlled self-revision — none of which
behaves like a deep-learning optimizer for the skill itself, and none of
which reliably improves over its starting point under feedback.

**SkillOpt treats the skill document as the trainable state of a frozen
agent**, and trains it with the discipline that makes weight-space
optimization reproducible. A separate optimizer model turns scored rollouts
into bounded add / delete / replace edits on a single skill document; a
candidate edit is accepted only when it strictly improves a held-out
validation score. A textual learning-rate budget, a rejected-edit buffer,
and an epoch-wise slow / meta update make skill training stable while
adding **zero inference-time model calls** at deployment.

The deployed artifact is a compact `best_skill.md` (typically 300–2,000
tokens) that runs against the unchanged target model. Across **six
benchmarks, seven target models, and three execution harnesses** (direct
chat, Codex CLI, Claude Code CLI), SkillOpt is best or tied-best on **all
52 evaluated (model, benchmark, harness) cells** and on GPT-5.5 lifts the
average no-skill accuracy by **+23.5 points in direct chat, +24.8 inside
the Codex agentic loop, and +19.1 inside Claude Code**. Optimized skill
artifacts transfer across model scales, between Codex and Claude Code
harnesses, and to nearby benchmarks without further optimization.

For the full method, ablations, and per-cell results see the [paper](https://arxiv.org/abs/2605.23904); for a visual walkthrough of the loop see the [project page](https://microsoft.github.io/SkillOpt/); for deeper API / backend / benchmark docs see [`docs/`](docs/).

## 🎬 Demo Video

https://github.com/user-attachments/assets/eb12d3bc-371c-467f-904d-91b61f339ed7

<p align="center">
  <a href="https://youtu.be/JUBMDTCiM0M"><b>▶ Watch the full demo on YouTube</b></a>
</p>

---

## Extensibility & WebUI

### Adding a new backend

A backend = a chat / exec target (e.g. `openai_chat`, `claude_chat`,
`qwen_chat`, `minimax_chat`, `codex_exec`, `claude_code_exec`). See
[`docs/guide/new-backend.md`](docs/guide/new-backend.md) for the full
contract; in short you add a `skillopt/model/<name>_backend.py` module,
register it in `skillopt/model/common.py` + `backend_config.py`, and wire
it through the router in `skillopt/model/__init__.py`. `qwen_backend.py`
and `minimax_backend.py` are good templates.

### Adding a new benchmark

A benchmark = a `skillopt/envs/<name>/` package with a `dataloader.py`, a
`rollout.py`, and an `initial.md` seed skill. See
[`docs/guide/new-benchmark.md`](docs/guide/new-benchmark.md) for the full
contract; the simplest reference is `skillopt/envs/searchqa/`.

### WebUI

Launch the monitoring dashboard (optional):

```bash
pip install -e ".[webui]"
python -m skillopt_webui.app
```

| Flag | Default | Description |
|---|---|---|
| `--port` | 7860 | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--share` | off | Create a public Gradio share link |

---

## Citation

```bibtex
@misc{yang2026skilloptexecutivestrategyselfevolving,
      title={SkillOpt: Executive Strategy for Self-Evolving Agent Skills}, 
      author={Yifan Yang and Ziyang Gong and Weiquan Huang and Qihao Yang and Ziwei Zhou and Zisu Huang and Yan Li and Xuemei Gao and Qi Dai and Bei Liu and Kai Qiu and Yuqing Yang and Dongdong Chen and Xue Yang and Chong Luo},
      year={2026},
      eprint={2605.23904},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.23904}
}
```
