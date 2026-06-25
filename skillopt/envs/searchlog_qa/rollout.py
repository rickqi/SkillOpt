"""SearchLogQA rollout — 企业文档搜索 QA 执行器。"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.envs.searchlog_qa.evaluator import evaluate
from skillopt.model import chat_target


_SYSTEM_TEMPLATE = """## 技能指令
{skill}

## 任务说明
你是企业文档搜索助手。根据提供的搜索上下文回答用户问题。
- 回答必须基于上下文中的信息，不要编造
- 如果上下文不足以回答问题，明确说明
- 将最终答案放在 <answer>...</answer> 标签中"""

_USER_TEMPLATE = """## 搜索上下文
{context}

## 用户问题
{question}

请给出你的回答："""


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    max_completion_tokens: int = 16384,
    exec_timeout: int = 120,
    **_kw,
) -> dict:
    """执行单条 QA 并评分。

    Parameters
    ----------
    item : dict
        包含 id, question, context, gold_answer。
    out_root : str
        输出根目录。
    skill_content : str
        当前技能文档。
    """
    item_id = str(item["id"])
    question = item.get("question", "")
    context = item.get("context", "")
    gold_answer = item.get("gold_answer", "")

    result = {
        "id": item_id,
        "question": question,
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "gold_answer": gold_answer,
        "response": "",
        "fail_reason": "",
        "task_description": question[:200],
        "task_type": item.get("domain", "searchlog_qa"),
        "n_turns": 1,
        "agent_ok": False,
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        system = _SYSTEM_TEMPLATE.format(skill=skill_content.strip() or "(尚无学习到的规则)")
        # 截断过长的上下文
        ctx = context[:8000] if len(context) > 8000 else context
        user = _USER_TEMPLATE.format(context=ctx, question=question)

        t0 = time.time()
        response, _ = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=3,
            stage="rollout",
            timeout=exec_timeout,
        )
        elapsed = time.time() - t0

        result["response"] = response
        result["agent_ok"] = True

        # 评分
        eval_result = evaluate(response, item)
        result["predicted_answer"] = eval_result["predicted_answer"]
        result["hard"] = eval_result["hard"]
        result["soft"] = eval_result["soft"]

        if not result["hard"]:
            result["fail_reason"] = (
                f"soft={eval_result['soft']:.2f}: "
                f"predicted={eval_result['predicted_answer'][:100]!r}"
            )

        # 保存轨迹（Reflect 阶段读取）
        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "hard": result["hard"],
                        "soft": result["soft"],
                        "predicted_answer": result["predicted_answer"],
                        "gold_answer": result["gold_answer"],
                        "elapsed_s": round(elapsed, 2),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)

    except Exception as e:
        result["fail_reason"] = f"error: {type(e).__name__}: {e}"

    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    workers: int = 8,
    max_completion_tokens: int = 16384,
    exec_timeout: int = 120,
    **_kw,
) -> list[dict]:
    """并行批量执行 QA。

    支持断点续传：已完成的条目会跳过。
    """
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    # 断点续传
    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    existing.append(r)
                except Exception:
                    pass

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if not pending:
        correct = sum(1 for r in existing if r.get("hard"))
        print(f"    [rollout] 全部 {len(existing)} 条已完成 (acc={correct/len(existing):.3f})")
        return existing

    total = len(existing) + len(pending)
    correct_count = sum(1 for r in existing if r.get("hard"))
    if existing:
        print(f"    [rollout] 续传: {len(existing)}/{total} 已完成")

    results = list(existing)

    with open(results_path, "a", encoding="utf-8") as outf:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    process_one,
                    it,
                    out_root,
                    skill_content,
                    max_completion_tokens,
                    exec_timeout,
                ): it
                for it in pending
            }

            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as exc:
                    item = futs[fut]
                    res = {
                        "id": str(item["id"]),
                        "question": item.get("question", ""),
                        "hard": 0,
                        "soft": 0.0,
                        "fail_reason": f"error: {type(exc).__name__}: {exc}",
                    }
                results.append(res)
                if res.get("hard"):
                    correct_count += 1
                acc = correct_count / len(results) if results else 0
                print(
                    f"    [rollout] {len(results)}/{total} "
                    f"(acc={acc:.3f}) id={res['id'][:30]} "
                    f"hard={res.get('hard', '?')} soft={res.get('soft', '?')}",
                    flush=True,
                )
                outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                outf.flush()

    return results
