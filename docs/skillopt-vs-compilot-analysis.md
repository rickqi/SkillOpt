# SkillOpt vs ComPilot — LLM Loop 设计对比分析

> 对比 SkillOpt (ReflACT) 与 ComPilot (arXiv 2511.00592) 两种 LLM 闭环优化方案的设计思路

---

## 一、概述

| 维度 | SkillOpt | ComPilot |
|---|---|---|
| **全称** | SkillOpt: Executive Strategy for Self-Evolving Agent Skills | ComPilot: Agentic Auto-Scheduling for LLM-Guided Loop Optimization |
| **论文** | [arXiv 2605.23904](https://arxiv.org/abs/2605.23904) | [arXiv 2511.00592](https://arxiv.org/abs/2511.00592) |
| **机构** | Microsoft | NYU Abu Dhabi |
| **优化目标** | Agent 技能文档（自然语言 Markdown） | 循环嵌套代码（C/C++ loop nest） |
| **迭代方式** | 固定 epochs × steps 结构化训练 | 开放式对话，LLM 自主决定何时停止 |
| **零样本** | ✅ 无微调 | ✅ 无微调 |

---

## 二、闭环架构对比

### SkillOpt — 6 阶段结构化训练 Loop

```
Epoch 1..N:
  Step 1..M:
    ① Rollout    → Target LLM 用当前 Skill 执行任务，收集 (score, trajectory)
    ② Reflect    → Optimizer LLM 分析轨迹，生成 edit patches (minibatch M=4)
    ③ Aggregate  → 分层合并多个 patches 为一个统一 patch
    ④ Select     → 按 edit_budget 排序裁剪 (gradient clipping 类比)
    ⑤ Update     → 将 edits 应用到 Skill 文档 (append/insert/replace/delete)
    ⑥ Gate       → 在 held-out val 集评估，严格提升才 ACCEPT，否则 REJECT
  
  Slow Update → 前后 epoch 对比 20 条相同样本，注入高层指导 (EMA 类比)
  Meta Skill  → 生成跨 epoch 策略记忆 (optimizer state 类比)
```

**设计特点**：
- 结构化的深度学习训练范式（forward pass → gradient → SGD step → validation）
- 双模型架构：Optimizer LLM（反思） + Target LLM（执行）
- Gate 机制确保单调不退化
- 产物是紧凑的 `.md` 文件（300-2000 tokens），零推理开销

### ComPilot — 开放式对话 Loop

```
Context Init:
  ① System Prompt → 定义 9 种变换原语 + 输出格式
  ② 展示 Loop Nest → 匿名化 + 标注 comp_ID
  ③ LLM 分析 → Chain-of-Thought

Iteration 1..T:
  ④ LLM 提议变换序列 → <schedule>...</schedule> 标签内
  ⑤ Compiler 检查合法性 → 5 类反馈 (valid/invalid/illegal/crash/runnable)
  ⑥ 生成 + 执行代码 → 测量 speedup/slowdown
  ⑦ LLM 读取全对话历史 → 调整策略 → 提议下一轮

停止条件: LLM 发出 stop 命令 或 达到 T_max
Multi-Run: 从头重启 K=5 次，取最佳
```

**设计特点**：
- 单模型同时承担探索和反思
- 对话历史作为 episodic memory (in-context learning)
- 无 Gate 验证机制，靠 Multi-Run 兜底
- 产物是可运行的优化代码

---

## 三、关键设计维度逐项对比

### 3.1 反馈信号

| | SkillOpt | ComPilot |
|---|---|---|
| **信号类型** | hard (0/1) + soft (0-1) | 5 类：valid/invalid/illegal/crash/runnable + speedup |
| **信号精度** | 粗粒度二元/连续 | 细粒度分类 + 精确数值 |
| **失败信息** | fail_reason 字符串 | 具体错误类型 + 原因 |
| **对 LLM 的价值** | 需 Optimizer LLM 自行推断 | 直接告知哪个环节出错 |

### 3.2 记忆机制

| | SkillOpt | ComPilot |
|---|---|---|
| **短期记忆** | 当前 step 的 rejection_context / step_buffer | 完整对话历史 |
| **长期记忆** | Meta-Skill（跨 epoch 策略记忆，2-3K chars） | ❌ 无（每次 Multi-Run 从零开始） |
| **正则化** | Slow Update（epoch 边界纵向对比注入） | ❌ 无 |
| **记忆效率** | 紧凑（2-3K chars meta-skill） | 冗余（完整对话，token 消耗非线性增长） |

### 3.3 防止退化

| | SkillOpt | ComPilot |
|---|---|---|
| **机制** | Gate：held-out val 集，严格提升才接受 | Multi-Run：跑 K 次取最佳 |
| **保证** | ✅ 单调不退化 | ❌ 每次 run 可能退化，靠 best-of-K 缓解 |
| **数据利用** | val 集独立于 train | 无独立验证集 |
| **鲁棒性** | Gate 强制门控 → 安全 | 概率性 → 需要足够多的 K |

### 3.4 探索 vs 利用

| | SkillOpt | ComPilot |
|---|---|---|
| **探索** | epoch 级别 seed shuffle + cosine LR decay | Multi-Run 随机重启 |
| **利用** | LR scheduler 逐步降低 edit budget | LLM 自主决定何时停止 |
| **局部最优** | Gate + Slow Update 帮助跳出 | Multi-Run 覆盖（但每次 run 可能陷入） |
| **过早停止** | ❌ 不存在（固定 step 数） | ⚠️ 常见问题，需 framework 推动 |

### 3.5 效率

| | SkillOpt | ComPilot |
|---|---|---|
| **API 调用** | Rollout(batch) + Reflect(minibatch) + Aggregate + Gate | 每轮 1 次 LLM 调用（但含全量历史） |
| **Token 消耗** | 线性增长（每 step 独立） | 非线性增长（对话历史膨胀） |
| **部署开销** | 零（产物是 .md 文件） | 需要编译器 + 执行环境 |
| **单次训练时间** | ~10 分钟（25 样本 × 4 epochs） | ~9 分钟（30 iterations） |
| **并行化** | ✅ 多 worker 并行 rollout + reflect | ❌ 串行对话 |

---

## 四、各自的优势与局限

### SkillOpt 优势
1. **Gate 机制**独特且关键 — 确保每次修改都是正向的
2. **双模型架构** — Optimizer 和 Target 可以不同（强模型反思，弱模型执行）
3. **跨模型迁移** — 优化好的 Skill 可以直接给不同模型用
4. **固定结构** — 不会过早停止，训练过程可预测
5. **产物极简** — 一个 .md 文件，零部署开销

### SkillOpt 局限
1. **反馈粗粒度** — 只有 hard/soft + fail_reason 字符串
2. **依赖 JSON 解析** — 非 OpenAI 模型可能输出格式不标准
3. **需要 held-out 验证集** — 小数据场景下 val 集噪音大
4. **训练成本固定** — 即使 Skill 已收敛也会跑完所有 epochs

### ComPilot 优势
1. **反馈细粒度** — 5 类错误 + 精确 speedup 数值
2. **探索灵活** — LLM 自主决定策略，不受固定 step 约束
3. **Multi-Run 鲁棒** — 随机重启覆盖搜索空间
4. **编译器兜底** — 合法性由编译器保证，不依赖 LLM

### ComPilot 局限
1. **无 Gate** — 单次 run 可能退化，靠 Multi-Run 概率兜底
2. **过早停止** — LLM 保守倾向，需额外机制推动
3. **Token 浪费** — 对话历史非线性膨胀，~64% 提议无效
4. **无长期记忆** — 每次 Multi-Run 从零开始
5. **部署重** — 需要编译器 + 执行环境

---

## 五、对比矩阵（总览）

```
                结构化程度          记忆机制           防退化
                ←──────────→    ←───────────→    ←──────────→
SkillOpt    ████████████░     ████████████░     ████████████░  (Gate)
ComPilot    ████░░░░░░░░     ███░░░░░░░░░     ██░░░░░░░░░░  (best-of-K)

                反馈精度           Token效率         部署轻量
                ←──────────→    ←───────────→    ←──────────→
SkillOpt    ████░░░░░░░░     ████████████░     ████████████░
ComPilot    ████████████░     ███░░░░░░░░░     ███░░░░░░░░░

                过早停止风险       探索覆盖
                ←──────────→    ←──────────→
SkillOpt    ██░░░░░░░░░░     ██████████░░  (epoch shuffle + LR decay)
ComPilot    ████████████░     ████████████  (Multi-Run 随机重启)
```

---

## 六、下一步建议方案

### 6.1 短期优化（基于 SkillOpt 现有框架）

**A. 细粒度反馈分类（借鉴 ComPilot）**

当前 SkillOpt 的 Reflect 只有 `analyst_error` 和 `analyst_success` 两类。可增加失败分类：

```yaml
# 在 reflect prompt 中增加失败分类指令
失败类型:
  1. SKILL_DEFECT     → 技能规则缺失/错误 → 正常 body edit
  2. EXECUTION_LAPSE  → 有规则但未执行 → appendix note (已有)
  3. AMBIGUOUS_QUERY  → 问题本身模糊 → 不处理
  4. CONTEXT_INSUFFICIENT → 上下文不足 → 不处理
```

对应修改 `skillopt/envs/searchlog_qa/evaluator.py` 增加 fail_reason 分类。

**B. Multi-Run Ensemble（借鉴 ComPilot best-of-K）**

```bash
# 跑 N 次独立训练，取最优 Skill
for i in 1 2 3 4 5; do
  python scripts/train.py --config configs/searchlog_qa/default.yaml --seed $i
done
# 在 test 集上评估所有 best_skill.md，选最优
```

可用 `scripts/eval_only.py` 批量评估。

**C. 自适应 Early Stopping**

当前跑满 4 epochs。可增加：连续 N 个 step 无 ACCEPT 时提前结束 epoch：

```yaml
optimizer:
  early_stop_patience: 3  # 连续 3 步无 accept 则提前结束 epoch
```

### 6.2 中期扩展

**D. 对话式 Reflect（借鉴 ComPilot in-context）**

当前 Reflect 每次独立调用。可改为多轮对话：第一次分析 → 得到初步 patch → 第二次基于第一次结果精炼：

```
Reflect Round 1: 分析 4 条失败轨迹 → 提出 3 个 edit
Reflect Round 2: 基于 Round 1 的 edit + 编译器反馈 → 修正为 2 个 edit
```

对应修改 `max_analyst_rounds` 参数（已有，目前默认 3）。

**E. 集成 LiteLLM 作为默认后端**

当前已支持通过 `SKILLOPT_EXTRA_BODY` 集成第三方 LLM。可进一步：
- 在 `skillopt/model/` 增加 `litellm_backend.py` 作为一等后端
- 支持模型自动 fallback（主模型超时 → 备用模型）

### 6.3 长期方向

**F. 跨任务 Skill 蒸馏**

类似 ComPilot 的 best-of-K 但跨不同 benchmark：
- 在 SearchQA 上训练 Skill A
- 在 OfficeQA 上训练 Skill B
- 用 Optimizer LLM 合并 A+B 的精华 → 通用 Skill C
- 验证 C 在 DocVQA 上的迁移效果

**G. LLM-as-Judge 自动评分**

当前评分依赖 token F1（有 gold_answer 时）或人工。可引入 LLM Judge：
- 用强模型（如 deepseek-chat）对比预测与 reference
- 输出 0-1 分数 + 理由
- 替换当前的 token F1 评分

---

## 七、实测对比（同一数据、同一模型）

### 实验设置

- **数据**：`data/searchlog_qa/`（train=25, val=8, test=9）
- **模型**：DeepSeek API `deepseek-chat`（同一模型用于两种方法）
- **初始Skill**：同一份 `initial.md`（40 chars 空模板）

### SkillOpt 结果（ReflACT 6阶段训练）

```
Epochs: 4    Steps: 16    Accepts: 5    Rejects: 8    Skips: 3
Best val soft: 0.6449 (Step 12)
Test hard:     0.4444  (+44.4pp vs baseline 0.0000)
Wall time:     585s
Tokens:        1,784,669
Calls:         424
Skill size:    4,223 chars
```

**训练曲线**：
```
Step 1:  soft 0.020 → 0.026  ACCEPT
Step 3:  soft 0.026 → 0.020  REJECT
Step 4:  soft 0.020 → 0.070  ACCEPT
Step 9:  soft 0.070 → 0.103  ACCEPT
Step 10: soft 0.103 → 0.497  ACCEPT
Step 12: soft 0.497 → 0.645  ACCEPT
```

### ComPilot 结果（闭环对话 Multi-Run K=3）

```
Runs:  3    Max iter/run: 12    Stop condition: LLM自主 / 3次连续无编辑
Best-of-3 val soft: 0.1352 (Run 1, Iter 4)
Test hard:          0.0000
Wall time:          451s
Avg iterations/run: 5.0
Skill size:         ~1,500 chars
```

**各 Run 详情**：
```
Run 1:  9 iters  best_soft=0.1352  模式: 0.02→0.04→0.05→0.14→0.03→0.09→0.03→0.08→0.04→STOP
        明显的震荡——每次improvement后紧跟regression，LLM无法稳定收敛

Run 2:  4 iters  best_soft=0.0604  模式: 0.03→0.05→0.06→0.05→STOP
        小幅提升后停止，且2次LLM输出无有效编辑

Run 3:  2 iters  best_soft=0.0401  模式: 0.00→0.04→STOP
        几乎无提升即停止
```

### 关键差异分析

| 指标 | SkillOpt | ComPilot | 差距 |
|---|---|---|---|
| **Test hard** | **0.4444** | 0.0000 | — |
| **Best val soft** | **0.6449** | 0.1352 | **4.8x** |
| Accepts/Total | 5/16 (31%) | — | — |
| 收敛性 | 单调上升(有Gate) | 剧烈震荡 | — |
| 过早停止 | 不存在(固定16步) | 2/3 runs提前STOP | — |
| Wall time | 585s | 451s | 1.3x |
| 代码量 | 12个文件 | 1个文件 | 12x |
| 并行化 | ✅ Rollout/Reflect并行 | ❌ 串行对话 | — |

### 失败原因分析

**ComPilot 为什么在这个任务上不如 SkillOpt？**

1. **无 Gate 导致震荡**：Run 1 的最佳值 0.1352 出现在 Iter 4，但 LLM 无法识别这是最优值，继续修改导致退化到 0.04。相比之下，SkillOpt 的 Gate 将最佳值 0.6449 锁定了下来。

2. **单模型瓶颈**：同一 LLM 既要"分析问题"又要"提出方案"——相当于没有专门的 Optimizer。SkillOpt 用强模型做 Reflect（minibatch 分析），模型做 Execute（单条任务），分工明确。

3. **上下文窗口限制**：ComPilot 用对话历史做记忆，但在 8 条样本的反馈信息超过 6000 tokens 后，早期策略信息被挤出窗口。SkillOpt 的 Meta-Skill 将策略精炼为 2-3K chars，高效利用上下文。

4. **无 Minibatch 视角**：ComPilot 每次只看 8 条的得分变化，没有跨样本的对比分析。SkillOpt 的 Reflect 将 4 条轨迹放在一起分析（minibatch M=4），能发现系统性模式。

5. **ComPilot 的"过早停止"问题完全复现**：Run 2 在 4 次迭代后停止，Run 3 在 2 次后停止——这正是论文中描述的 premature stopping 问题。

### 两种方法适用场景

| | 适合场景 |
|---|---|
| **SkillOpt** | ✅ 训练数据25+条，需要稳定不退化，跨模型迁移，轻量部署 |
| **ComPilot** | ✅ 编译器类场景（有精确合法性反馈），允许 Multi-Run 覆盖退化 |
| **本任务结论** | **SkillOpt 显著优于 ComPilot**——Gate 是决定性优势 |

## 八、总结与建议

### 核心发现

SkillOpt 的 Gate 机制是区分两种方法最关键的因素。在没有 Gate 保护的情况下（ComPilot 模式），LLM 会在"探索更优解"和"误操作导致退化"之间反复震荡，最终难以稳定收敛。

### 下一步建议优先级

1. 🔴 **高优先**：细粒度失败分类（6.1-A）— 改动最小，收益最大
2. 🟡 **中优先**：Multi-Run Ensemble on SkillOpt（6.1-B）— 直接叠加到已有 SkillOpt 框架
3. 🟢 **长期**：LLM-as-Judge 评分（6.3-G）— 摆脱对 gold_answer 的依赖
4. ⚪ **已完成**：ComPilot 对比实验 — 验证了 SkillOpt 在此类任务上的优势
