import json
import random
import os
import sys
from collections import defaultdict
from pathlib import Path
import argparse

from src.output_optimizer import detect_task_type
from src.fireworks_client import chat
from src.optimizer.execution import execute_task
from src import validator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default=os.environ.get("EVAL_DATASET", str(Path(__file__).parent / "input" / "tasks.json")),
)
parser.add_argument("--difficulty", default=os.environ.get("EVAL_DIFFICULTY"))
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()
DATA_PATH = Path(args.dataset)
random.seed(args.seed)

# ── GLM-5.2 Judge ──
MODEL_JUDGE = "accounts/fireworks/models/glm-5p2"

def verify_with_glm(prompt: str, answer: str, task_type: str) -> dict:
    """Ask GLM-5.2 to verify an answer.  Returns {verdict, reason, tokens}.
    Verdict is 'correct', 'incorrect', or 'error' (on API failure)."""
    judge_prompt = (
        f"You are a strict answer checker. Given the task and the candidate answer, "
        f"decide if the answer is correct and complete.\n\n"
        f"Task:\n{prompt}\n\n"
        f"Candidate Answer:\n{answer}\n\n"
        f"Reply with EXACTLY one line: CORRECT or INCORRECT followed by a short reason."
    )
    try:
        resp = chat(
            model=MODEL_JUDGE,
            prompt=judge_prompt,
            max_tokens=60,
            system_prompt="You are a precise answer verifier. Output one line: CORRECT or INCORRECT with a brief reason.",
            extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
        )
        text = resp["text"].strip()
        verdict = "correct" if text.upper().startswith("CORRECT") else "incorrect"
        return {
            "verdict": verdict,
            "reason": text,
            "tokens": resp["total_tokens"]
        }
    except Exception as e:
        print(f"[JUDGE ERROR] GLM-5.2 verification failed: {e}")
        return {"verdict": "error", "reason": str(e), "tokens": 0}

if "ALLOWED_MODELS" in os.environ:
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    MODEL = next((m for m in models if "kimi" in m.lower()), models[-1] if models else "accounts/fireworks/models/kimi-k2p6")
else:
    MODEL = os.environ.get("MODEL", "accounts/fireworks/models/kimi-k2p6")


def sample_tasks(records, total_to_sample=40):
    """Sample tasks, ensuring all categories are represented evenly."""
    by_category = defaultdict(list)
    for r in records:
        by_category[r.get("category", "general")].append(r)
    
    categories = list(by_category.keys())
    per_cat = max(1, total_to_sample // len(categories))
    
    sampled = []
    for cat in categories:
        cat_records = by_category[cat]
        if len(cat_records) > per_cat:
            sampled.extend(random.sample(cat_records, per_cat))
        else:
            sampled.extend(cat_records)
            
    # If we still need more to reach exactly total_to_sample
    remaining = total_to_sample - len(sampled)
    if remaining > 0:
        unused = [r for r in records if r not in sampled]
        sampled.extend(random.sample(unused, min(remaining, len(unused))))
        
    random.shuffle(sampled)
    return sampled[:total_to_sample]


def main():
    print(f"Loading dataset from {DATA_PATH}...")
    with open(DATA_PATH, "r") as f:
        records = json.load(f)
        
    if args.difficulty:
        records = [r for r in records if r.get("difficulty_pool") == args.difficulty]
        
    # Sample exactly 40 entries, including all categories
    tasks = sample_tasks(records, 40)
    print(f"Sampled {len(tasks)} tasks across {len(set(t.get('category', 'unknown') for t in tasks))} categories.")
    print("=" * 80)
    
    total_tokens = 0
    judge_tokens = 0
    judge_results = {"correct": 0, "incorrect": 0, "error": 0}
    success_count = 0
    results = []
    factual_tasks = []
    codegen_tasks = []
    logic_tasks = []
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("task_id", task.get("id", f"task_{i}"))
        prompt = task["prompt"]
        print(f"\n--- TASK {i}/{len(tasks)} [{task_id}] ---")
        print(f"Category (Dataset): {task.get('category', 'unknown')}")
        
        task_type = detect_task_type(prompt)
        print(f"Detected Category (Heuristic): {task_type}")
        print(f"Prompt:\n{prompt}\n")
        
        model = MODEL
        print(f"[ROUTER] Using model: {model}")
        
        try:
            answer = execute_task(
                prompt=prompt,
                model=model,
                chat_fn=chat,
                extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
            )
            total_tokens += answer["total_tokens"]
            task_type = answer["optimization"]["task_type"]
            validation = answer["validation"]

            print(
                f"[API CALL] Model: {model} | Cap: {answer['optimization']['max_tokens']} "
                f"| Solver: {answer['solver_type']}"
            )
            print(
                f"[RESPONSE] Tokens: {answer['total_tokens']} "
                f"(prompt={answer['prompt_tokens']}, completion={answer['completion_tokens']}) "
                f"| Finish Reason: {answer.get('finish_reason', 'none')}"
            )
            print(f"[TEXT]\n{answer['text']}")

            if not validation.valid:
                print(f"[VALIDATION FAILED] Reason: {'; '.join(validation.errors)}.")
            else:
                print("[VALIDATION PASSED] Output looks good.")
                
            entry = {
                "task_id": task_id,
                "category_dataset": task.get("category", "unknown"),
                "category_detected": task_type,
                "prompt": prompt,
                "solver_type": answer["solver_type"],
                "model_or_solver": model,
                "tokens_used": answer["total_tokens"],
                "prompt_tokens": answer["prompt_tokens"],
                "completion_tokens": answer["completion_tokens"],
                "max_tokens": answer["optimization"]["max_tokens"],
                "router_confidence": answer["optimization"]["confidence"],
                "token_policy_reason": answer["optimization"]["metadata"]["token_policy_reason"],
                "output": answer["text"],
                "validation_passed": validation.valid,
                "validation_errors": validation.errors,
            }
            jv = verify_with_glm(prompt, answer["text"], task_type)
            entry["judge_verdict"] = jv["verdict"]
            entry["judge_reason"] = jv["reason"]
            entry["judge_tokens"] = jv["tokens"]
            judge_tokens += jv["tokens"]
            judge_results[jv["verdict"]] += 1
            if jv["verdict"] == "incorrect":
                print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
            else:
                print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
            results.append(entry)
                
            success_count += 1
                
        except Exception as e:
            print(f"[ERROR] API Call failed: {e}")
            results.append({
                "task_id": task_id,
                "category_dataset": task.get("category", "unknown"),
                "category_detected": task_type,
                "prompt": prompt,
                "solver_type": "error",
                "model_or_solver": model,
                "tokens_used": 0,
                "output": str(e),
                "validation_passed": False
            })
            
        print("\n<EOT>\n" + "=" * 80)
        
    if factual_tasks:
        print(f"\n--- PROCESSING {len(factual_tasks)} BUNDLED FACTUAL TASKS ---")
        bundled_prompt = "Please answer the following factual knowledge questions briefly. Separate each answer with exactly the string '|||' (three pipes).\n\n"
        for idx, item in enumerate(factual_tasks, 1):
            bundled_prompt += f"Question {idx}: {item['prompt']}\n"
            
        model = MODEL
        print(f"[API CALL BUNDLED] Model: {model} | Reasoning: None")
        try:
            answer = chat(
                model=model,
                prompt=bundled_prompt,
                max_tokens=2000,
                system_prompt="You are a precise and helpful assistant. Answer each question briefly and directly. You MUST separate each answer with exactly '|||'. Do not include 'Question x:' in your response.",
                extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
            )
            total_tokens += answer["total_tokens"]
            
            print(f"[RESPONSE BUNDLED] Tokens: {answer['total_tokens']}")
            print(f"[TEXT BUNDLED]\n{answer['text']}")
            
            answers = [a.strip() for a in answer["text"].split("|||")]
            
            tokens_per_task = answer["total_tokens"] // len(factual_tasks)
            remainder = answer["total_tokens"] % len(factual_tasks)
            
            if len(answers) != len(factual_tasks):
                print(f"[WARN] Expected {len(factual_tasks)} answers, got {len(answers)}. Will process what we can.")
                
            for idx, item in enumerate(factual_tasks):
                t_id = item["task_id"]
                t_prompt = item["prompt"]
                t_type = item["task_type"]
                cat = item["task"].get("category", "unknown")
                
                ans_text = answers[idx] if idx < len(answers) else "Failed to parse bundled answer."
                
                t_tokens = tokens_per_task + (1 if idx < remainder else 0)
                
                # Check validation (e.g. for empty or invalid)
                ok, reason = validator.validate(t_type, t_prompt, ans_text, "stop")
                if not ok or not ans_text:
                    ans_text = f"Validation failed: {reason}"
                    
                entry = {
                    "task_id": t_id,
                    "category_dataset": cat,
                    "category_detected": t_type,
                    "prompt": t_prompt,
                    "solver_type": "api_bundled",
                    "model_or_solver": f"{model} (bundled)",
                    "tokens_used": t_tokens,
                    "output": ans_text,
                    "validation_passed": ok
                }
                
                jv = verify_with_glm(t_prompt, ans_text, t_type)
                entry["judge_verdict"] = jv["verdict"]
                entry["judge_reason"] = jv["reason"]
                entry["judge_tokens"] = jv["tokens"]
                judge_tokens += jv["tokens"]
                judge_results[jv["verdict"]] += 1
                
                if jv["verdict"] == "incorrect":
                    print(f"[JUDGE ⚠] {t_id} incorrect: {jv['reason']}")
                else:
                    print(f"[JUDGE ✓] {t_id} verified: {jv['reason']}")
                    
                results.append(entry)
                success_count += 1
                
        except Exception as e:
            print(f"[ERROR] Bundled API Call failed: {e}")
            for item in factual_tasks:
                results.append({
                    "task_id": item["task_id"],
                    "category_dataset": item["task"].get("category", "unknown"),
                    "category_detected": item["task_type"],
                    "prompt": item["prompt"],
                    "solver_type": "error",
                    "model_or_solver": model,
                    "tokens_used": 0,
                    "output": str(e),
                    "validation_passed": False
                })
        print("\n<EOT>\n" + "=" * 80)

    if codegen_tasks:
        print(f"\n--- PROCESSING {len(codegen_tasks)} BUNDLED CODEGEN TASKS ---")
        bundled_prompt = "Please write the requested code for the following tasks. Separate each response with exactly the string '|||' (three pipes).\n\n"
        for idx, item in enumerate(codegen_tasks, 1):
            bundled_prompt += f"Task {idx}: {item['prompt']}\n\n"
            
        model = MODEL
        print(f"[API CALL BUNDLED] Model: {model} | Reasoning: None")
        try:
            answer = chat(
                model=model,
                prompt=bundled_prompt,
                max_tokens=4000,
                system_prompt="You are a precise coding assistant. Output ONLY the requested code for each task, minified. You MUST separate each code response with exactly '|||'. Do not include 'Task x:' in your response.",
                extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
            )
            total_tokens += answer["total_tokens"]
            
            print(f"[RESPONSE BUNDLED CODEGEN] Tokens: {answer['total_tokens']}")
            print(f"[TEXT BUNDLED CODEGEN]\n{answer['text']}")
            
            answers = [a.strip() for a in answer["text"].split("|||")]
            
            tokens_per_task = answer["total_tokens"] // len(codegen_tasks)
            remainder = answer["total_tokens"] % len(codegen_tasks)
            
            if len(answers) != len(codegen_tasks):
                print(f"[WARN] Expected {len(codegen_tasks)} answers, got {len(answers)}. Will process what we can.")
                
            for idx, item in enumerate(codegen_tasks):
                t_id = item["task_id"]
                t_prompt = item["prompt"]
                t_type = item["task_type"]
                cat = item["task"].get("category", "unknown")
                
                ans_text = answers[idx] if idx < len(answers) else "Failed to parse bundled answer."
                
                t_tokens = tokens_per_task + (1 if idx < remainder else 0)
                
                # Check validation
                ok, reason = validator.validate(t_type, t_prompt, ans_text, "stop")
                if not ok or not ans_text:
                    ans_text = f"Validation failed: {reason}"
                    
                entry = {
                    "task_id": t_id,
                    "category_dataset": cat,
                    "category_detected": t_type,
                    "prompt": t_prompt,
                    "solver_type": "api_bundled",
                    "model_or_solver": f"{model} (bundled)",
                    "tokens_used": t_tokens,
                    "output": ans_text,
                    "validation_passed": ok
                }
                
                jv = verify_with_glm(t_prompt, ans_text, t_type)
                entry["judge_verdict"] = jv["verdict"]
                entry["judge_reason"] = jv["reason"]
                entry["judge_tokens"] = jv["tokens"]
                judge_tokens += jv["tokens"]
                judge_results[jv["verdict"]] += 1
                
                if jv["verdict"] == "incorrect":
                    print(f"[JUDGE ⚠] {t_id} incorrect: {jv['reason']}")
                else:
                    print(f"[JUDGE ✓] {t_id} verified: {jv['reason']}")
                    
                results.append(entry)
                success_count += 1
                
        except Exception as e:
            print(f"[ERROR] Bundled API Call failed: {e}")
            for item in codegen_tasks:
                results.append({
                    "task_id": item["task_id"],
                    "category_dataset": item["task"].get("category", "unknown"),
                    "category_detected": item["task_type"],
                    "prompt": item["prompt"],
                    "solver_type": "error",
                    "model_or_solver": model,
                    "tokens_used": 0,
                    "output": str(e),
                    "validation_passed": False
                })
        print("\n<EOT>\n" + "=" * 80)

    if logic_tasks:
        print(f"\n--- PROCESSING {len(logic_tasks)} BUNDLED LOGIC TASKS ---")
        bundled_prompt = "Please solve the following logic puzzles. Separate each answer with exactly the string '|||' (three pipes).\n\n"
        for idx, item in enumerate(logic_tasks, 1):
            bundled_prompt += f"Puzzle {idx}: {item['prompt']}\n\n"
            
        model = MODEL
        print(f"[API CALL BUNDLED] Model: {model} | Reasoning: None")
        try:
            answer = chat(
                model=model,
                prompt=bundled_prompt,
                max_tokens=2000,
                system_prompt="You are a precise logic solver. Answer concisely. You MUST separate each answer with exactly '|||'. Do not include 'Puzzle x:' in your response.",
                extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
            )
            total_tokens += answer["total_tokens"]
            
            print(f"[RESPONSE BUNDLED LOGIC] Tokens: {answer['total_tokens']}")
            print(f"[TEXT BUNDLED LOGIC]\n{answer['text']}")
            
            answers = [a.strip() for a in answer["text"].split("|||")]
            
            tokens_per_task = answer["total_tokens"] // len(logic_tasks)
            remainder = answer["total_tokens"] % len(logic_tasks)
            
            if len(answers) != len(logic_tasks):
                print(f"[WARN] Expected {len(logic_tasks)} answers, got {len(answers)}. Will process what we can.")
                
            for idx, item in enumerate(logic_tasks):
                t_id = item["task_id"]
                t_prompt = item["prompt"]
                t_type = item["task_type"]
                cat = item["task"].get("category", "unknown")
                
                ans_text = answers[idx] if idx < len(answers) else "Failed to parse bundled answer."
                
                t_tokens = tokens_per_task + (1 if idx < remainder else 0)
                
                # Check validation
                ok, reason = validator.validate(t_type, t_prompt, ans_text, "stop")
                if not ok or not ans_text:
                    ans_text = f"Validation failed: {reason}\nOutput was: {ans_text}"
                    
                entry = {
                    "task_id": t_id,
                    "category_dataset": cat,
                    "category_detected": t_type,
                    "prompt": t_prompt,
                    "solver_type": "api_bundled",
                    "model_or_solver": f"{model} (bundled)",
                    "tokens_used": t_tokens,
                    "output": ans_text,
                    "validation_passed": ok
                }
                
                jv = verify_with_glm(t_prompt, ans_text, t_type)
                entry["judge_verdict"] = jv["verdict"]
                entry["judge_reason"] = jv["reason"]
                entry["judge_tokens"] = jv["tokens"]
                judge_tokens += jv["tokens"]
                judge_results[jv["verdict"]] += 1
                
                if jv["verdict"] == "incorrect":
                    print(f"[JUDGE ⚠] {t_id} incorrect: {jv['reason']}")
                else:
                    print(f"[JUDGE ✓] {t_id} verified: {jv['reason']}")
                    
                results.append(entry)
                success_count += 1
                
        except Exception as e:
            print(f"[ERROR] Bundled API Call failed: {e}")
            for item in logic_tasks:
                results.append({
                    "task_id": item["task_id"],
                    "category_dataset": item["task"].get("category", "unknown"),
                    "category_detected": item["task_type"],
                    "prompt": item["prompt"],
                    "solver_type": "error",
                    "model_or_solver": model,
                    "tokens_used": 0,
                    "output": str(e),
                    "validation_passed": False
                })
        print("\n<EOT>\n" + "=" * 80)
        
    category_breakdown = {}
    detected_category_breakdown = {}
    total_local = 0
    for r in results:
        cat = r.get("category_dataset", "unknown")
        det_cat = r.get("category_detected", "fallback")
        
        if cat not in category_breakdown:
            category_breakdown[cat] = {
                "total_questions": 0,
                "local_count": 0,
                "api_count": 0,
                "tokens_used": 0,
                "judge_correct": 0,
                "judge_total": 0
            }
        if det_cat not in detected_category_breakdown:
            detected_category_breakdown[det_cat] = {
                "total_questions": 0,
                "local_count": 0,
                "api_count": 0,
                "tokens_used": 0,
                "judge_correct": 0,
                "judge_total": 0
            }
            
        category_breakdown[cat]["total_questions"] += 1
        category_breakdown[cat]["tokens_used"] += r.get("tokens_used", 0)
        
        detected_category_breakdown[det_cat]["total_questions"] += 1
        detected_category_breakdown[det_cat]["tokens_used"] += r.get("tokens_used", 0)
        
        if r.get("judge_verdict") in ("correct", "incorrect"):
            category_breakdown[cat]["judge_total"] += 1
            detected_category_breakdown[det_cat]["judge_total"] += 1
            if r.get("judge_verdict") == "correct":
                category_breakdown[cat]["judge_correct"] += 1
                detected_category_breakdown[det_cat]["judge_correct"] += 1
                
        if r.get("solver_type") == "local":
            category_breakdown[cat]["local_count"] += 1
            detected_category_breakdown[det_cat]["local_count"] += 1
            total_local += 1
        else:
            category_breakdown[cat]["api_count"] += 1
            detected_category_breakdown[det_cat]["api_count"] += 1

    from datetime import datetime
    timestamp_str = datetime.now().strftime("%d%m%Y%H%M")
    out_path = Path(__file__).parent / "results" / f"eval_{timestamp_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_judged = judge_results["correct"] + judge_results["incorrect"]
    overall_accuracy = judge_results["correct"] / total_judged * 100 if total_judged > 0 else 0.0

    output_data = {
        "summary": {
            "total_tasks": len(tasks),
            "success_count": success_count,
            "total_local_tasks": total_local,
            "total_api_tasks": len(tasks) - total_local,
            "total_api_tokens": total_tokens,
            "approximate_score": total_tokens / len(tasks) * 19 if len(tasks) > 0 else 0,
            "judge_accuracy_pct": round(overall_accuracy, 1),
            "judge_results": judge_results,
            "judge_tokens": judge_tokens,
            "category_breakdown": category_breakdown,
            "detected_category_breakdown": detected_category_breakdown
        },
        "results": results
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed evaluation results to {out_path.resolve()}")

    print("\n" + "#" * 80)
    print(f"EVALUATION COMPLETE")
    print(f"Total API Tokens Used: {total_tokens}")
    print(f"Tasks Successfully Processed: {success_count}/{len(tasks)}")
    print(f"Total Tasks Handled Locally (0 tokens): {total_local}/{len(tasks)}")
    print(f"Approximate score: {total_tokens/len(tasks)*19:.2f}")
    print("-" * 80)
    print("Category Breakdown (Dataset Categories):")
    for cat, stats in sorted(category_breakdown.items()):
        jt = stats.get("judge_total", 0)
        jc = stats.get("judge_correct", 0)
        acc_str = f"{jc/jt*100:.0f}%" if jt > 0 else "N/A"
        print(f"  • {cat:<26} | Total: {stats['total_questions']:<2} | Local: {stats['local_count']:<2} | API: {stats['api_count']:<2} | Tokens: {stats['tokens_used']:<4} | Acc: {acc_str} ({jc}/{jt})")
    print("-" * 80)
    print("Detected Category Breakdown (Heuristics - includes uncategorized 'fallback' tasks):")
    for cat, stats in sorted(detected_category_breakdown.items()):
        jt = stats.get("judge_total", 0)
        jc = stats.get("judge_correct", 0)
        acc_str = f"{jc/jt*100:.0f}%" if jt > 0 else "N/A"
        print(f"  • {cat:<26} | Total: {stats['total_questions']:<2} | Local: {stats['local_count']:<2} | API: {stats['api_count']:<2} | Tokens: {stats['tokens_used']:<4} | Acc: {acc_str} ({jc}/{jt})")
    print("-" * 80)
    print(f"GLM-5.2 Judge Results: ✓ {judge_results['correct']} correct | ⚠ {judge_results['incorrect']} incorrect | ✗ {judge_results['error']} errors")
    print(f"GLM-5.2 Judge Accuracy: {overall_accuracy:.1f}% ({judge_results['correct']}/{total_judged})")
    print(f"GLM-5.2 Judge Tokens Used: {judge_tokens} (not counted in solver score)")
    print("#" * 80)


if __name__ == "__main__":
    main()
