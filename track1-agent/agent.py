"""Container entrypoint for AMD Developer Hackathon Act II, Track 1.

Matches the judging harness contract exactly:
  - reads tasks from /input/tasks.json: [{"task_id": "...", "prompt": "..."}]
  - writes /output/results.json: [{"task_id": "...", "answer": "..."}]
  - exits 0 on success, non-zero on failure
  - all answer-generating calls go through FIREWORKS_BASE_URL with a model
    from ALLOWED_MODELS - the router itself runs locally and costs zero tokens

The optimizer decides whether work can be handled locally or needs the single
Fireworks model. ``ALLOWED_MODELS`` is supplied by the judging harness; when it
contains Kimi, Kimi is preferred. Local routing itself consumes zero tokens.
"""
import json
import os
import sys
from pathlib import Path

from src.fireworks_client import chat
from src.optimizer.execution import solve_tasks, structured_log

INPUT_PATH = Path(os.environ.get("TASK_INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("TASK_OUTPUT_PATH", "/output/results.json"))

if "ALLOWED_MODELS" in os.environ:
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    MODEL = next((m for m in models if "kimi" in m.lower()), models[-1] if models else "accounts/fireworks/models/kimi-k2p6")
else:
    MODEL = os.environ.get("MODEL", "accounts/fireworks/models/kimi-k2p6")


def main():
    tasks = json.loads(INPUT_PATH.read_text())
    results = []
    solved = solve_tasks(
        tasks=tasks,
        model=MODEL,
        chat_fn=chat,
        extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
    )
    for answer in solved["results"]:
        if answer.get("error"):
            print(f"[ERROR] API handling failed for task {answer['task_id']}: {answer['error']}", file=sys.stderr)
        validation = answer.get("validation")
        if validation is not None and not validation.valid:
            reasons = "; ".join(validation.errors)
            print(f"Validation failed for task {answer['task_id']} ({reasons}).", file=sys.stderr)
        if (
            os.environ.get("OPTIMIZER_STRUCTURED_LOG", "").lower() in {"1", "true", "yes"}
            and validation is not None
        ):
            print(structured_log(answer), file=sys.stderr)
        results.append({"task_id": answer["task_id"], "answer": answer["text"]})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(
        f"Wrote {len(results)} results to {OUTPUT_PATH}. Total tokens: {solved['total_tokens']}. "
        f"API calls: {solved['individual_calls']} individual + {solved['bundle_calls']} bundled; "
        f"bundle fallbacks: {solved['bundle_fallbacks']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"agent failed: {e}", file=sys.stderr)
        sys.exit(1)
