"""ComPilot-style Skill Optimization Trainer.

Implements ComPilot's closed-loop dialogue approach (arXiv 2511.00592)
applied to the same searchlog_qa domain used by SkillOpt.

Key differences from SkillOpt:
  - Single LLM (no separate optimizer/target)
  - Open-ended dialogue (not fixed 6-stage pipeline)
  - In-context episodic memory (not Meta-Skill cross-epoch)
  - Multi-Run best-of-K (not Gate validation)
  - Fine-grained feedback (5 categories, not binary hard/soft)

Usage:
  python src/compilot_train.py --config configs/compilot/default.yaml
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_dotenv():
    """Load .env into process environment."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _get_llm_client():
    """Create LLM client from environment."""
    from openai import OpenAI
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://api.deepseek.com/v1")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    return OpenAI(base_url=endpoint.rstrip("/"), api_key=api_key)


def _load_extra_body() -> dict | None:
    raw = os.environ.get("SKILLOPT_EXTRA_BODY", "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ComPilot System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """## Role
You are an AI skill optimization agent. Your task is to iteratively improve a skill document (system prompt) that guides an LLM to answer enterprise document search queries.

## Process
For each iteration:
1. I will show you the current skill document and a set of test examples with their scores
2. You analyze what needs improvement based on the feedback
3. You propose edits to the skill document inside <edit>...</edit> tags
4. I will apply your edits, re-evaluate, and report back the new scores
5. You continue refining until you believe the skill is optimal, then output <stop/>

## Edit Format
Propose edits as a JSON array. Each edit must have:
- "op": one of "append", "replace", "delete", "insert_after"
- "content": the text to add/replace with (for delete, leave empty)
- "target": for replace/delete/insert_after, the exact text to match in the current skill
- "reasoning": why this edit should improve performance

Example:
<edit>
[
  {"op": "replace", "target": "old instruction", "content": "new instruction", "reasoning": "The old instruction was too vague"},
  {"op": "append", "content": "## New Rule\\nAlways do X when Y happens", "reasoning": "Addresses common failure pattern"}
]
</edit>

## Feedback Format
After each iteration, you will receive:
- Overall score: hard=X/N soft=Y (higher is better)
- Per-example breakdown showing which improved, regressed, or stayed unchanged
- For failures: the question, predicted answer, and gold answer

## Strategy Guidelines
1. Make SMALL, targeted edits (1-3 per iteration). Large rewrites often break working rules.
2. When examples regress after your edit, revert the problematic change in the next iteration.
3. Address the most common failure patterns first.
4. If score stops improving after 3 consecutive attempts, consider stopping.
5. Preserve rules that work well — only modify what needs fixing.

## Response Format
Always respond with one of:
- <edit>[JSON array of edits]</edit> followed by brief reasoning
- <stop/> when you believe the skill cannot be further improved"""


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator (reuse from searchlog_qa)
# ═══════════════════════════════════════════════════════════════════════════════

from skillopt.envs.searchlog_qa.rollout import _SYSTEM_TEMPLATE, _USER_TEMPLATE


def _evaluate_skill(skill_content: str, items: list[dict], client, model: str, extra_body: dict | None) -> dict:
    """Evaluate skill on a set of items using same prompts as SkillOpt rollout."""
    from skillopt.envs.searchlog_qa.evaluator import evaluate

    results = []
    total_hard = 0

    for item in items:
        try:
            system = _SYSTEM_TEMPLATE.format(skill=skill_content.strip() or "(empty)")
            ctx = item.get("context", "")[:8000]
            user = _USER_TEMPLATE.format(context=ctx, question=item["question"])

            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=2048,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = client.chat.completions.create(**kwargs)
            response = resp.choices[0].message.content or ""

            eval_r = evaluate(response, item)
            eval_r["id"] = item["id"]
            eval_r["question"] = item["question"][:80]
            eval_r["response"] = response
            results.append(eval_r)
            total_hard += eval_r["hard"]
        except Exception as e:
            results.append({"id": item.get("id", "?"), "hard": 0, "soft": 0.0, "error": str(e)})

    n = len(items)
    return {
        "hard": total_hard / n if n else 0,
        "soft": sum(r.get("soft", 0) for r in results) / n if n else 0,
        "hard_count": total_hard,
        "total": n,
        "per_item": results,
    }


def _format_feedback(prev_eval: dict, curr_eval: dict, train_items: list[dict]) -> str:
    """Generate ComPilot-style feedback message."""
    lines = []

    prev_hard = prev_eval.get("hard_count", 0)
    curr_hard = curr_eval.get("hard_count", 0)
    prev_soft = prev_eval.get("soft", 0)
    curr_soft = curr_eval.get("soft", 0)
    total = curr_eval.get("total", 0)

    # Category determination (ComPilot-style)
    if prev_eval.get("total", 0) == 0:
        lines.append("## Evaluation: Initial Baseline")
        lines.append(f"\nCurrent score: hard={curr_hard}/{total} ({curr_hard/total:.2%} if total > 0 else 0)  soft={curr_soft:.4f}\n")
    elif curr_soft > prev_soft + 0.005:
        lines.append(f"## Evaluation: SUCCESS_IMPROVED (soft: {prev_soft:.3f} -> {curr_soft:.3f}, hard: {prev_hard}/{total} -> {curr_hard}/{total})")
    elif abs(curr_soft - prev_soft) <= 0.005:
        lines.append(f"## Evaluation: SUCCESS_UNCHANGED (soft: {curr_soft:.3f}, hard: {curr_hard}/{total})")
    else:
        lines.append(f"## Evaluation: REGRESSION (soft: {prev_soft:.3f} -> {curr_soft:.3f}, hard: {prev_hard}/{total} -> {curr_hard}/{total})")

    lines.append(f"\nCurrent score: hard={curr_hard}/{total} ({curr_hard/total:.2%})  soft={curr_soft:.4f}\n")

    # Per-item breakdown — show failures and changes
    curr_map = {r["id"]: r for r in curr_eval.get("per_item", [])}
    prev_map = {r["id"]: r for r in prev_eval.get("per_item", [])}

    improved, regressed, unchanged_fail, unchanged_pass = [], [], [], []
    for item_id, cr in curr_map.items():
        pr = prev_map.get(item_id, {})
        if cr.get("hard"):
            if pr.get("hard"):
                unchanged_pass.append(cr)
            else:
                improved.append(cr)
        else:
            if pr and pr.get("hard"):
                regressed.append(cr)
            else:
                unchanged_fail.append(cr)

    if improved:
        lines.append(f"### ✅ Improved ({len(improved)} examples)")
        for r in improved[:3]:
            lines.append(f"- [{r['id'][:20]}] Q: {r.get('question', '')[:60]}")
    if regressed:
        lines.append(f"\n### ⚠️ Regressed ({len(regressed)} examples)")
        for r in regressed[:3]:
            lines.append(f"- [{r['id'][:20]}] Q: {r.get('question', '')[:60]}")
            lines.append(f"  Predicted: {r.get('predicted_answer', '')[:100]}")
            lines.append(f"  Gold: {r.get('gold_answer', '')[:100]}")
    if unchanged_fail:
        lines.append(f"\n### ❌ Still Failing ({len(unchanged_fail)} examples)")
        for r in unchanged_fail[:5]:
            lines.append(f"- [{r['id'][:20]}] Q: {r.get('question', '')[:60]}")
            lines.append(f"  Predicted: {r.get('predicted_answer', '')[:80]}")
            lines.append(f"  Gold: {r.get('gold_answer', '')[:80]}")

    return "\n".join(lines)


def _extract_edits(response: str) -> list[dict] | None:
    """Extract edits from LLM response. Returns None if stop, [] if no edits found."""
    import re

    if "<stop" in response.lower():
        return None  # signal to stop

    # Try <edit> tag first
    m = re.search(r"<edit>(.*?)</edit>", response, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            edits = json.loads(m.group(1).strip())
            if isinstance(edits, dict):
                edits = [edits]
            return edits if isinstance(edits, list) else []
        except json.JSONDecodeError:
            pass

    # Fallback: try to find any JSON array in the response
    m2 = re.search(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
    if m2:
        try:
            edits = json.loads(m2.group(0))
            if isinstance(edits, list) and len(edits) > 0:
                return edits
        except json.JSONDecodeError:
            pass

    return []


def _apply_edits(skill: str, edits: list[dict]) -> tuple[str, list[str]]:
    """Apply edits to skill document. Returns (new_skill, report)."""
    report = []
    for i, edit in enumerate(edits):
        op = edit.get("op", "append")
        content = edit.get("content", "")
        target = edit.get("target", "")
        reasoning = edit.get("reasoning", "")

        try:
            if op == "append":
                skill = skill.rstrip() + "\n\n" + content + "\n"
                report.append(f"edit_{i}: append OK ({reasoning[:50]})")
            elif op == "replace" and target and target in skill:
                skill = skill.replace(target, content, 1)
                report.append(f"edit_{i}: replace OK ({reasoning[:50]})")
            elif op == "delete" and target and target in skill:
                skill = skill.replace(target, "", 1)
                report.append(f"edit_{i}: delete OK ({reasoning[:50]})")
            elif op == "insert_after" and target and target in skill:
                idx = skill.index(target) + len(target)
                nl = skill.find("\n", idx)
                insert_at = nl + 1 if nl != -1 else len(skill)
                skill = skill[:insert_at] + "\n" + content + "\n" + skill[insert_at:]
                report.append(f"edit_{i}: insert_after OK ({reasoning[:50]})")
            else:
                report.append(f"edit_{i}: {op} SKIPPED — target not found or invalid")
        except Exception as e:
            report.append(f"edit_{i}: {op} ERROR — {e}")

    return skill, report


# ═══════════════════════════════════════════════════════════════════════════════
# Single Dialogue Run
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_dialogue(
    train_items: list[dict],
    skill_content: str,
    model: str,
    client,
    extra_body: dict | None,
    max_iter: int = 15,
    eval_sample_size: int = 8,
    verbose: bool = True,
) -> dict:
    """Run one ComPilot-style optimization dialogue.

    Returns dict with: best_skill, best_score, history, iterations
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    current_skill = skill_content
    history = []
    best_skill = skill_content
    best_score = {"hard": 0.0, "soft": 0.0}

    # Sample eval items (ComPilot uses a fixed set per dialogue for consistent feedback)
    import random
    rng = random.Random(42)
    eval_items = rng.sample(train_items, min(eval_sample_size, len(train_items)))

    # Evaluate baseline
    prev_eval = _evaluate_skill(current_skill, eval_items, client, model, extra_body)
    history.append({"iter": 0, "hard": prev_eval["hard"], "soft": prev_eval["soft"], "skill": current_skill})
    best_skill = current_skill
    best_score = {"hard": prev_eval["hard"], "soft": prev_eval["soft"]}

    if verbose:
        print(f"  [Iter 0] baseline: hard={prev_eval['hard_count']}/{prev_eval['total']} soft={prev_eval['soft']:.4f}")

    consecutive_no_edits = 0
    for iteration in range(1, max_iter + 1):
        # prev_iter_eval tracks evaluation before this iteration's edit
        prev_iter_eval = prev_eval

        # Build user message with current state
        if iteration == 1:
            feedback = _format_feedback({"total": 0, "hard_count": 0, "soft": 0.0, "per_item": []}, prev_eval, train_items)
        else:
            feedback = _format_feedback(prev_iter_eval, prev_eval, train_items)
        user_msg = (
            f"## Current Skill\n```markdown\n{current_skill[:6000]}\n```\n\n"
            f"{feedback}\n\n"
            f"## Instruction\n"
            f"Analyze the feedback above. Propose targeted edits in <edit>...</edit> tags "
            f"to improve the skill. Focus on the most common failure patterns. "
            f"If no further improvement seems possible, output <stop/>."
        )

        messages.append({"role": "user", "content": user_msg})

        # Call LLM
        try:
            kwargs = dict(model=model, messages=messages, max_completion_tokens=4096)
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            response = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [Iter {iteration}] LLM error: {e}")
            break

        messages.append({"role": "assistant", "content": response})

        # Check for stop
        if "<stop" in response.lower():
            if verbose:
                print(f"  [Iter {iteration}] LLM issued STOP (score={prev_eval['soft']:.4f})")
            break

        # Extract and apply edits
        edits = _extract_edits(response)
        if edits is None:
            break  # stop signal
        if not edits:
            consecutive_no_edits += 1
            if verbose:
                print(f"  [Iter {iteration}] no valid edits parsed (#{consecutive_no_edits}), continuing...")
            if consecutive_no_edits >= 3:
                if verbose:
                    print(f"  [Iter {iteration}] 3 consecutive no-edit attempts, stopping dialogue")
                break
            messages.append({"role": "user", "content": "No valid edits were found. Please propose at least one specific edit to improve the skill, or output <stop/> if truly done."})
            continue
        consecutive_no_edits = 0  # reset on successful parse

        new_skill, edit_report = _apply_edits(current_skill, edits)

        # Evaluate new skill
        curr_eval = _evaluate_skill(new_skill, eval_items, client, model, extra_body)
        history.append({"iter": iteration, "hard": curr_eval["hard"], "soft": curr_eval["soft"],
                        "skill": new_skill, "edits": len(edits), "report": edit_report})

        if verbose:
            delta = curr_eval["soft"] - prev_eval["soft"]
            sign = "+" if delta > 0 else ""
            print(f"  [Iter {iteration}] {len(edits)} edits -> hard={curr_eval['hard_count']}/{curr_eval['total']} "
                  f"soft={curr_eval['soft']:.4f} ({sign}{delta:.4f})  [{'; '.join(edit_report)}]")

        # Track best
        if curr_eval["soft"] > best_score["soft"]:
            best_skill = new_skill
            best_score = {"hard": curr_eval["hard"], "soft": curr_eval["soft"]}

        # Adaptation: if this edit made things worse, revert
        if curr_eval["soft"] < prev_eval["soft"] - 0.01:
            if verbose:
                print(f"    -> Regression detected, keeping for LLM self-correction")
            # Don't revert current_skill — let LLM see the regression and fix in next iteration
            # (ComPilot keeps the feedback and lets LLM self-correct)

        current_skill = new_skill
        prev_eval = curr_eval

        # Trim message history to prevent context overflow
        if len(messages) > 20:
            messages = [messages[0]] + messages[-16:]

    return {
        "best_skill": best_skill,
        "best_score": best_score,
        "final_skill": current_skill,
        "final_score": {"hard": prev_eval["hard"], "soft": prev_eval["soft"]},
        "history": history,
        "iterations": len(history) - 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ComPilot-style Skill Optimization")
    parser.add_argument("--config", type=str, default="configs/compilot/default.yaml")
    parser.add_argument("--model", type=str, default=os.environ.get("COMPILOT_MODEL", "deepseek-chat"))
    parser.add_argument("--num-runs", type=int, default=5, help="Multi-Run: K independent dialogues")
    parser.add_argument("--max-iter", type=int, default=15, help="Max iterations per dialogue")
    parser.add_argument("--eval-sample", type=int, default=8, help="Examples to evaluate per iteration")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    _load_dotenv()

    # Output dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or str(_PROJECT_ROOT / "outputs" / f"compilot_searchlog_qa_{args.model}_{ts}")
    os.makedirs(out_root, exist_ok=True)

    # Load data
    data_dir = _PROJECT_ROOT / "data" / "searchlog_qa"
    train_items = json.load(open(data_dir / "train" / "items.json", encoding="utf-8"))
    val_items = json.load(open(data_dir / "val" / "items.json", encoding="utf-8"))
    test_items = json.load(open(data_dir / "test" / "items.json", encoding="utf-8"))

    # Initial skill
    skill_init_path = _PROJECT_ROOT / "skillopt" / "envs" / "searchlog_qa" / "skills" / "initial.md"
    initial_skill = open(skill_init_path, encoding="utf-8").read() if skill_init_path.exists() else "(empty)"

    # Setup LLM
    client = _get_llm_client()
    extra_body = _load_extra_body()

    print(f"\n{'='*60}")
    print(f"  ComPilot — Agentic Skill Optimization (Multi-Run K={args.num_runs})")
    print(f"{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Train:      {len(train_items)} items")
    print(f"  Val:        {len(val_items)} items")
    print(f"  Test:       {len(test_items)} items")
    print(f"  Max iter:   {args.max_iter}")
    print(f"  Output:     {out_root}")
    print(f"{'='*60}\n")

    # ═══ Multi-Run ═══
    all_runs = []
    for run_idx in range(1, args.num_runs + 1):
        print(f"\n── Run {run_idx}/{args.num_runs} " + "─" * 45)
        t0 = time.time()
        result = run_single_dialogue(
            train_items=train_items,
            skill_content=initial_skill,
            model=args.model,
            client=client,
            extra_body=extra_body,
            max_iter=args.max_iter,
            eval_sample_size=args.eval_sample,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        result["run"] = run_idx
        result["wall_time"] = elapsed
        all_runs.append(result)

        print(f"  Run {run_idx} done: best_soft={result['best_score']['soft']:.4f}, "
              f"iters={result['iterations']}, time={elapsed:.0f}s")

        # Save run result
        run_dir = os.path.join(out_root, f"run_{run_idx:02d}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "best_skill.md"), "w", encoding="utf-8-sig") as f:
            f.write(result["best_skill"])
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in h.items() if k != "skill"} for h in result["history"]],
                      f, ensure_ascii=False, indent=2)

    # ═══ Select Best (ComPilot best-of-K) ═══
    best_run = max(all_runs, key=lambda r: r["best_score"]["soft"])
    print(f"\n{'='*60}")
    print(f"  Best-of-{args.num_runs}: Run {best_run['run']} (soft={best_run['best_score']['soft']:.4f})")
    print(f"{'='*60}\n")

    # ═══ Test Evaluation ═══
    print("  Evaluating on Test set...")
    baseline_eval = _evaluate_skill(initial_skill, test_items, client, args.model, extra_body)
    best_eval = _evaluate_skill(best_run["best_skill"], test_items, client, args.model, extra_body)

    # Also evaluate all runs to get best-of-K test score
    best_test_score = 0.0
    best_test_skill = best_run["best_skill"]
    for run in all_runs:
        test_eval = _evaluate_skill(run["best_skill"], test_items, client, args.model, extra_body)
        run["test_score"] = test_eval
        if test_eval["hard"] > best_test_score:
            best_test_score = test_eval["hard"]
            best_test_skill = run["best_skill"]

    # Save best on test
    with open(os.path.join(out_root, "best_skill_test.md"), "w", encoding="utf-8-sig") as f:
        f.write(best_test_skill)

    # ═══ Summary ═══
    print(f"\n{'='*60}")
    print(f"  ComPilot Final Results")
    print(f"{'='*60}")
    print(f"  Baseline (no skill):     hard={baseline_eval['hard_count']}/{baseline_eval['total']} "
          f"soft={baseline_eval['soft']:.4f}")
    print(f"  Best-of-{args.num_runs} on val (run {best_run['run']}): hard={best_run['best_score']['hard']:.3f} "
          f"soft={best_run['best_score']['soft']:.4f}")
    print(f"  Best-of-{args.num_runs} on test:       hard={best_test_score:.4f} "
          f"soft={best_eval['soft']:.4f}")
    print(f"  Avg iterations/run:      {sum(r['iterations'] for r in all_runs)/len(all_runs):.1f}")
    print(f"  Total wall time:         {sum(r['wall_time'] for r in all_runs):.0f}s")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "method": "ComPilot",
        "model": args.model,
        "num_runs": args.num_runs,
        "max_iter": args.max_iter,
        "baseline_test": {"hard": baseline_eval["hard"], "soft": baseline_eval["soft"]},
        "best_val": {"run": best_run["run"], "hard": best_run["best_score"]["hard"], "soft": best_run["best_score"]["soft"]},
        "best_test": {"hard": best_test_score, "soft": best_eval["soft"]},
        "runs": [{"run": r["run"], "iterations": r["iterations"], "best_soft": r["best_score"]["soft"],
                  "wall_time": r["wall_time"], "test_hard": r.get("test_score", {}).get("hard", 0)}
                 for r in all_runs],
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved to: {out_root}")


if __name__ == "__main__":
    main()
