"""
Train a BERT sequence classifier on the SST-2 sentiment dataset (HuggingFace Trainer).

Run: python bert_sentiment_analysis.py --help
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

# SST-2: 0 = negative, 1 = positive
LABEL_NAMES = ("negative", "positive")
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

QUICK_TRAIN_CAP = 4096
QUICK_EPOCHS = 1


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="weighted")),
    }


def tokenize_sst2_batch(
    batch: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    return tokenizer(
        batch["sentence"],
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def plot_loss_curves(log_history: list[dict], out_path: Path) -> None:
    train_step, train_loss = [], []
    eval_step, eval_loss = [], []

    for row in log_history:
        if "loss" in row and "eval_loss" not in row:
            train_step.append(int(row.get("step", 0)))
            train_loss.append(float(row["loss"]))
        if "eval_loss" in row:
            eval_step.append(int(row.get("step", row.get("epoch", 0))))
            eval_loss.append(float(row["eval_loss"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_step:
        ax.plot(train_step, train_loss, label="Train loss", alpha=0.85)
    if eval_step:
        ax.plot(eval_step, eval_loss, "o-", label="Validation loss", alpha=0.85)
    ax.set_xlabel("Global step (eval logged at checkpoint steps)")
    ax.set_ylabel("Loss")
    ax.set_title("SST-2 fine-tuning — train vs validation loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss plot to {out_path}")


def print_example_predictions(
    trainer: Trainer,
    validation_rows,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    num_samples: int,
) -> None:
    model = trainer.model
    device = next(model.parameters()).device
    n = min(num_samples, len(validation_rows))

    print("\n--- Sample predictions (validation, first rows) ---")
    model.eval()
    for i in range(n):
        sentence = validation_rows[i]["sentence"]
        true_id = int(validation_rows[i]["label"])
        batch = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        pred_id = int(logits.argmax(dim=-1).item())
        print(f"\nText: {sentence!r}")
        print(f"Gold:     {ID2LABEL[true_id]} ({true_id})")
        print(f"Predicted:{ID2LABEL[pred_id]} ({pred_id})")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    base = script_dir()
    p = argparse.ArgumentParser(
        description="Fine-tune a BERT-style classifier on SST-2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model-name", help="HF model id (e.g. bert-base-uncased).")
    p.add_argument("--dataset", help="datasets.load_dataset(...) id.")
    p.add_argument("--output-dir", type=Path, help="Checkpoints and Trainer output.")
    p.add_argument("--logs-dir", type=Path, help="Logging directory for Trainer.")
    p.add_argument("--plots-dir", type=Path, help="Where to write loss plot and metrics JSON.")

    p.add_argument(
        "--quick",
        action="store_true",
        help=f"Use {QUICK_EPOCHS} epoch(s) and at most {QUICK_TRAIN_CAP} training rows (unless --max-train-samples is set).",
    )
    p.add_argument("--epochs", type=int, help="Number of training epochs.")
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        metavar="N",
        help="Shuffle then keep only N training rows (omit for full train split).",
    )
    p.add_argument("--shuffle-seed", type=int, help="Seed for shuffling before subsampling.")
    p.add_argument("--per-device-train-batch-size", type=int)
    p.add_argument("--per-device-eval-batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--checkpoint-every-steps", type=int, help="Run validation and save a checkpoint every N steps.")
    p.add_argument("--save-total-limit", type=int, help="Rotate old checkpoints; best checkpoint is still kept.")
    p.add_argument(
        "--logging-steps",
        type=int,
        default=None,
        help="Trainer logging interval (default: min(50, checkpoint-every-steps)).",
    )
    p.add_argument(
        "--metric-for-best-model",
        choices=("eval_f1", "eval_accuracy", "eval_loss"),
        help="Which validation metric picks the best checkpoint.",
    )
    p.add_argument(
        "--greater-is-better",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force comparison direction (default: high is better except eval_loss).",
    )
    p.add_argument("--max-length", type=int, help="Tokenizer truncation length.")
    p.add_argument("--num-example-predictions", type=int, help="How many validation rows to print after training.")
    p.add_argument("--seed", type=int, help="RNG seed for Trainer.")
    p.add_argument(
        "--save-safetensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save weights as safetensors when supported.",
    )
    p.add_argument("--report-to", help="Trainer integration (e.g. none, tensorboard).")

    p.set_defaults(
        model_name="bert-base-uncased",
        dataset="sst2",
        output_dir=base / "results",
        logs_dir=base / "logs",
        plots_dir=base / "plots",
        epochs=3,
        shuffle_seed=42,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        checkpoint_every_steps=5,
        save_total_limit=30,
        metric_for_best_model="eval_f1",
        max_length=128,
        num_example_predictions=5,
        seed=42,
        report_to="none",
    )
    return p


@dataclass(frozen=True)
class ResolvedRun:
    """Single place for all values that actually drive training (after --quick rules)."""

    model_name: str
    dataset: str
    output_dir: Path
    logs_dir: Path
    plots_dir: Path
    epochs: int
    max_train_samples: int | None
    shuffle_seed: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    checkpoint_every_steps: int
    save_total_limit: int
    logging_steps: int
    metric_for_best_model: str
    greater_is_better: bool
    max_length: int
    num_example_predictions: int
    seed: int
    save_safetensors: bool
    report_to: str


def resolve_run(ns: argparse.Namespace) -> ResolvedRun:
    epochs = QUICK_EPOCHS if ns.quick else ns.epochs
    max_train = ns.max_train_samples
    if ns.quick and ns.max_train_samples is None:
        max_train = QUICK_TRAIN_CAP

    ckpt_every = max(1, ns.checkpoint_every_steps)
    log_every = ns.logging_steps if ns.logging_steps is not None else min(50, ckpt_every)

    if ns.greater_is_better is not None:
        greater = ns.greater_is_better
    else:
        greater = ns.metric_for_best_model != "eval_loss"

    return ResolvedRun(
        model_name=ns.model_name,
        dataset=ns.dataset,
        output_dir=Path(ns.output_dir).resolve(),
        logs_dir=Path(ns.logs_dir).resolve(),
        plots_dir=Path(ns.plots_dir).resolve(),
        epochs=epochs,
        max_train_samples=max_train,
        shuffle_seed=ns.shuffle_seed,
        per_device_train_batch_size=ns.per_device_train_batch_size,
        per_device_eval_batch_size=ns.per_device_eval_batch_size,
        learning_rate=ns.learning_rate,
        checkpoint_every_steps=ckpt_every,
        save_total_limit=ns.save_total_limit,
        logging_steps=log_every,
        metric_for_best_model=ns.metric_for_best_model,
        greater_is_better=greater,
        max_length=ns.max_length,
        num_example_predictions=ns.num_example_predictions,
        seed=ns.seed,
        save_safetensors=ns.save_safetensors,
        report_to=ns.report_to,
    )


def build_training_arguments(cfg: ResolvedRun) -> TrainingArguments:
    # eval_steps must match save_steps when load_best_model_at_end=True.
    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        eval_strategy="steps",
        eval_steps=cfg.checkpoint_every_steps,
        save_strategy="steps",
        save_steps=cfg.checkpoint_every_steps,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_dir=str(cfg.logs_dir),
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        save_total_limit=cfg.save_total_limit,
        save_safetensors=cfg.save_safetensors,
        seed=cfg.seed,
    )


def main() -> None:
    ns = build_parser().parse_args()
    cfg = resolve_run(ns)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    dataset: DatasetDict = load_dataset(cfg.dataset)
    print("\n=== Data ===")
    print(f"Dataset: {cfg.dataset}")
    print(f"Train rows:     {len(dataset['train']):,}")
    print(f"Validation rows:{len(dataset['validation']):,}")

    tokenize = partial(tokenize_sst2_batch, tokenizer=tokenizer, max_length=cfg.max_length)
    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_split = tokenized["train"]
    if cfg.max_train_samples is not None:
        n = min(cfg.max_train_samples, len(train_split))
        train_split = train_split.shuffle(seed=cfg.shuffle_seed).select(range(n))
        print(f"\nTraining on {n:,} rows (subsampled).")

    if ns.quick:
        print(f"\n(--quick → {cfg.epochs} epoch(s), train cap as above.)\n")

    training_args = build_training_arguments(cfg)

    print("\n=== Train config ===")
    print(f"  epochs:              {cfg.epochs}")
    print(f"  batch train / eval:  {cfg.per_device_train_batch_size} / {cfg.per_device_eval_batch_size}")
    print(f"  learning_rate:       {cfg.learning_rate}")
    print(f"  max_length:          {cfg.max_length}")
    print(f"  checkpoint / eval:   every {cfg.checkpoint_every_steps} steps")
    print(f"  best checkpoint by:  {cfg.metric_for_best_model} (greater_is_better={cfg.greater_is_better})")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()

    results = trainer.evaluate()
    print("\n=== Validation ===")
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.plots_dir / "eval_metrics.json"
    serializable = {
        k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in results.items()
    }
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path}")

    plot_loss_curves(trainer.state.log_history, cfg.plots_dir / "loss_curves.png")

    print_example_predictions(
        trainer,
        dataset["validation"],
        tokenizer,
        max_length=cfg.max_length,
        num_samples=cfg.num_example_predictions,
    )


if __name__ == "__main__":
    main()
