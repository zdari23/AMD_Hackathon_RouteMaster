"""Predict easy/hard locally, with an optional fine-tuned DistilBERT model.

When the checkpoint and ML dependencies are available, the fine-tuned model is
used. Deployments without that private artifact fall back to the project's
deterministic router. Both paths are local and consume zero API tokens.
"""
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "router-distilbert"

_model = None
_tokenizer = None
_device = None


def _load():
    global _model, _tokenizer, _device
    if _model is not None:
        return
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(f"Router checkpoint not found: {CHECKPOINT_DIR}")

    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    _device = torch.device("cpu")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(CHECKPOINT_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(_device)
    _model.eval()


def _deterministic_predict(prompt: str) -> str:
    from src.optimizer.router import route_task

    task_type = route_task(prompt).task_type
    hard_categories = {"bug_fixing", "code_authoring", "logical_puzzles", "fallback"}
    return "hard" if task_type in hard_categories else "easy"


def predict_with_backend(prompt: str) -> tuple[str, str]:
    """Return ``(label, backend)`` without requiring optional ML packages."""
    try:
        _load()
        import torch

        enc = _tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(_device)
        with torch.no_grad():
            logits = _model(**enc).logits
        label_id = logits.argmax(dim=-1).item()
        return ("hard" if label_id == 1 else "easy"), "fine-tuned DistilBERT"
    except (FileNotFoundError, ImportError, OSError):
        return _deterministic_predict(prompt), "deterministic local fallback"


def predict(prompt: str) -> str:
    """Return ``easy`` or ``hard`` using the best available local router."""
    return predict_with_backend(prompt)[0]


if __name__ == "__main__":
    import sys
    print(predict(sys.argv[1] if len(sys.argv) > 1 else "What is 2 + 2?"))
