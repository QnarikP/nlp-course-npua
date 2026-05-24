"""
Fine-tune BERT on SST-2, or run inference on new text.

  python bert_sentiment_analysis.py --help
  python bert_sentiment_analysis.py --train-fraction 0.1 --no-auto-retry
  python bert_sentiment_analysis.py --predict --text "This film was great."
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

LABEL_NAMES = ("negative", "positive")
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

QUICK_TRAIN_CAP = 4096
QUICK_EPOCHS = 1
DEFAULT_MIN_EVAL_F1 = 0.90
DEFAULT_MAX_RETRIES = 5


def setup_file_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bert_sst2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


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


def plot_training_curves(log_history: list[dict], out_path: Path) -> None:
    train_step, train_loss = [], []
    eval_step, eval_loss, eval_f1, eval_acc = [], [], [], []

    for row in log_history:
        if "loss" in row and "eval_loss" not in row:
            train_step.append(int(row.get("step", 0)))
            train_loss.append(float(row["loss"]))
        if "eval_loss" in row:
            step = int(row.get("step", 0))
            eval_step.append(step)
            eval_loss.append(float(row["eval_loss"]))
            if "eval_f1" in row:
                eval_f1.append(float(row["eval_f1"]))
            if "eval_accuracy" in row:
                eval_acc.append(float(row["eval_accuracy"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if train_step:
        ax.plot(train_step, train_loss, label="Train loss", alpha=0.85)
    if eval_step:
        ax.plot(eval_step, eval_loss, "o-", label="Validation loss", alpha=0.85)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_title("SST-2 — loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axes[1]
    if eval_step and eval_f1:
        ax.plot(eval_step, eval_f1, "o-", label="Validation F1", color="green", alpha=0.85)
    if eval_step and eval_acc:
        ax.plot(eval_step, eval_acc, "s--", label="Validation accuracy", color="blue", alpha=0.7)
    ax.set_xlabel("Global step (end of each eval)")
    ax.set_ylabel("Score")
    ax.set_title("SST-2 — validation metrics")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_gold_label(raw: str) -> int:
    token = raw.strip().lower()
    if token in LABEL2ID:
        return LABEL2ID[token]
    if token in {"0", "neg", "negative"}:
        return 0
    if token in {"1", "pos", "positive"}:
        return 1
    raise ValueError(f"Unknown label {raw!r}; use positive/negative or 1/0.")


def load_gold_labels(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    labels = [parse_gold_label(line) for line in lines if line.strip()]
    return labels


def plot_inference_accuracy(*, correct: int, incorrect: int, accuracy: float, out_path: Path) -> None:
    total = correct + incorrect
    if total == 0:
        return

    correct_pct = 100.0 * correct / total
    incorrect_pct = 100.0 * incorrect / total

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    sizes = [correct, incorrect]
    colors = ["#2ecc71", "#e74c3c"]
    labels = [f"Correct\n{correct_pct:.1f}%", f"Incorrect\n{incorrect_pct:.1f}%"]
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title("Inference accuracy")

    ax = axes[1]
    bars = ax.bar(["Correct", "Incorrect"], [correct_pct, incorrect_pct], color=colors, width=0.55)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Accuracy: {accuracy * 100:.1f}% ({correct}/{total})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, pct in zip(bars, [correct_pct, incorrect_pct]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{pct:.1f}%", ha="center", fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    print("\n--- Sample predictions (validation) ---")
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
        print(f"Gold:      {ID2LABEL[true_id]} ({true_id})")
        print(f"Predicted: {ID2LABEL[pred_id]} ({pred_id})")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    base = script_dir()
    p = argparse.ArgumentParser(
        description="Fine-tune BERT on SST-2 (full train + monitoring + optional retries).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model-name")
    p.add_argument("--dataset")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--logs-dir", type=Path)
    p.add_argument("--plots-dir", type=Path)
    p.add_argument("--run-dir", type=Path, default=None, help="Per-attempt artifacts (metrics, plots, run log).")

    p.add_argument("--quick", action="store_true", help="1 epoch, capped train rows (smoke test only).")
    p.add_argument("--epochs", type=int)
    p.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        metavar="F",
        help="Use F of the train split after shuffle (e.g. 0.1 = 10%%). Ignored if --max-train-samples is set.",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap training rows after shuffle (overrides --train-fraction).",
    )
    p.add_argument("--shuffle-seed", type=int)
    p.add_argument("--per-device-train-batch-size", type=int)
    p.add_argument("--per-device-eval-batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--warmup-ratio", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument(
        "--eval-strategy",
        choices=("epoch", "steps"),
        default=None,
        help="Validation schedule (default: epoch for full run, steps for --quick).",
    )
    p.add_argument("--checkpoint-every-steps", type=int, help="Used when eval-strategy=steps.")
    p.add_argument("--save-total-limit", type=int)
    p.add_argument("--logging-steps", type=int, default=None)
    p.add_argument("--metric-for-best-model", choices=("eval_f1", "eval_accuracy", "eval_loss"))
    p.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=2, help="Epochs without improvement before stop.")
    p.add_argument("--max-length", type=int)
    p.add_argument("--num-example-predictions", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--report-to", default=None, help="none or tensorboard.")
    p.add_argument("--force-cpu", action="store_true", help="Ignore GPU even if available.")

    p.add_argument("--min-eval-f1", type=float, default=DEFAULT_MIN_EVAL_F1, help="Stop retries when val F1 reaches this.")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max training attempts (seeds 42, 43, ...).")
    p.add_argument("--no-auto-retry", action="store_true", help="Run once even if F1 is below --min-eval-f1.")

    infer = p.add_argument_group("inference (use instead of training)")
    infer.add_argument(
        "--predict",
        action="store_true",
        help="Load a checkpoint and classify --text / --text-file (no training).",
    )
    infer.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint folder (default: latest checkpoint-* in --output-dir).",
    )
    infer.add_argument("--text", action="append", default=[], help="Sentence to classify (repeatable).")
    infer.add_argument("--text-file", type=Path, default=None, help="One sentence per line.")
    infer.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Gold labels (one per line: positive/negative or 1/0). Same order as --text-file.",
    )
    infer.add_argument(
        "--accuracy-plot",
        type=Path,
        default=None,
        help="Save accuracy chart when --labels-file is set (default: plots/inference_accuracy.png).",
    )
    infer.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write prediction JSON (inference mode only).",
    )

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
        warmup_ratio=0.1,
        weight_decay=0.01,
        checkpoint_every_steps=500,
        save_total_limit=3,
        metric_for_best_model="eval_f1",
        max_length=128,
        num_example_predictions=5,
        seed=42,
        report_to="tensorboard",
    )
    return p


@dataclass(frozen=True)
class ResolvedRun:
    model_name: str
    dataset: str
    output_dir: Path
    logs_dir: Path
    plots_dir: Path
    run_dir: Path
    epochs: int
    max_train_samples: int | None
    shuffle_seed: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    eval_strategy: str
    checkpoint_every_steps: int
    save_total_limit: int
    logging_steps: int
    metric_for_best_model: str
    greater_is_better: bool
    max_length: int
    num_example_predictions: int
    seed: int
    report_to: str
    early_stopping_patience: int
    use_fp16: bool
    min_eval_f1: float
    max_retries: int
    auto_retry: bool
    attempt: int


def resolve_max_train_samples(
    ns: argparse.Namespace,
    *,
    full_train_size: int | None = None,
) -> int | None:
    """Pick how many training rows to use (None = full split)."""
    if ns.max_train_samples is not None:
        return ns.max_train_samples
    if ns.train_fraction is not None:
        if not 0 < ns.train_fraction <= 1:
            raise ValueError("--train-fraction must be in (0, 1], e.g. 0.1 for 10%")
        if full_train_size is None:
            raise ValueError("full_train_size required when using --train-fraction")
        return max(1, int(full_train_size * ns.train_fraction))
    if ns.quick:
        return QUICK_TRAIN_CAP
    return None


def resolve_run(ns: argparse.Namespace, *, attempt: int, base_seed: int) -> ResolvedRun:
    quick = ns.quick
    epochs = QUICK_EPOCHS if quick else ns.epochs
    max_train = ns.max_train_samples
    if quick and max_train is None:
        max_train = QUICK_TRAIN_CAP

    eval_strategy = ns.eval_strategy
    if eval_strategy is None:
        eval_strategy = "steps" if quick else "epoch"

    ckpt_every = max(1, ns.checkpoint_every_steps)
    log_every = ns.logging_steps if ns.logging_steps is not None else (10 if quick else 50)

    if ns.greater_is_better is not None:
        greater = ns.greater_is_better
    else:
        greater = ns.metric_for_best_model != "eval_loss"

    report_to = ns.report_to if ns.report_to is not None else ("none" if quick else "tensorboard")
    run_dir = Path(ns.run_dir).resolve() if ns.run_dir else (script_dir() / "runs" / f"attempt_{attempt:02d}")

    return ResolvedRun(
        model_name=ns.model_name,
        dataset=ns.dataset,
        output_dir=Path(ns.output_dir).resolve(),
        logs_dir=Path(ns.logs_dir).resolve(),
        plots_dir=Path(ns.plots_dir).resolve(),
        run_dir=run_dir,
        epochs=epochs,
        max_train_samples=max_train,
        shuffle_seed=ns.shuffle_seed,
        per_device_train_batch_size=ns.per_device_train_batch_size,
        per_device_eval_batch_size=ns.per_device_eval_batch_size,
        learning_rate=ns.learning_rate,
        warmup_ratio=ns.warmup_ratio,
        weight_decay=ns.weight_decay,
        eval_strategy=eval_strategy,
        checkpoint_every_steps=ckpt_every,
        save_total_limit=ns.save_total_limit,
        logging_steps=log_every,
        metric_for_best_model=ns.metric_for_best_model,
        greater_is_better=greater,
        max_length=ns.max_length,
        num_example_predictions=ns.num_example_predictions,
        seed=base_seed + attempt,
        report_to=report_to,
        early_stopping_patience=ns.early_stopping_patience,
        use_fp16=torch.cuda.is_available() and not ns.force_cpu,
        min_eval_f1=ns.min_eval_f1,
        max_retries=ns.max_retries,
        auto_retry=not ns.no_auto_retry and not quick,
        attempt=attempt,
    )


def build_training_arguments(cfg: ResolvedRun) -> TrainingArguments:
    eval_steps = cfg.checkpoint_every_steps if cfg.eval_strategy == "steps" else None
    save_steps = eval_steps

    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy=cfg.eval_strategy,
        eval_steps=eval_steps,
        save_strategy=cfg.eval_strategy,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_dir=str(cfg.logs_dir),
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        report_to=cfg.report_to,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        fp16=cfg.use_fp16,
        dataloader_num_workers=0,
        disable_tqdm=False,
    )


def build_trainer(
    *,
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizerBase,
    callbacks: list | None = None,
) -> Trainer:
    kwargs: dict = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    if callbacks:
        kwargs["callbacks"] = callbacks
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def resolve_checkpoint_dir(checkpoint: Path | None, output_dir: Path) -> Path:
    if checkpoint is not None:
        path = checkpoint.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    root = output_dir.resolve()
    checkpoints = sorted(
        root.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if checkpoints:
        return checkpoints[-1]
    if (root / "config.json").exists():
        return root
    raise FileNotFoundError(f"No checkpoint under {root}. Train first or pass --checkpoint.")


@torch.no_grad()
def predict_sentences(
    texts: list[str],
    *,
    checkpoint_dir: Path,
    device: torch.device,
    max_length: int,
) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)

    rows = []
    for i, text in enumerate(texts):
        pred_id = int(logits[i].argmax().item())
        rows.append(
            {
                "text": text,
                "label": ID2LABEL[pred_id],
                "label_id": pred_id,
                "confidence": float(probs[i, pred_id].item()),
                "prob_negative": float(probs[i, 0].item()),
                "prob_positive": float(probs[i, 1].item()),
            }
        )
    return rows


def run_inference(ns: argparse.Namespace) -> None:
    texts = list(ns.text or [])
    if ns.text_file:
        lines = Path(ns.text_file).read_text(encoding="utf-8").splitlines()
        texts.extend(line.strip() for line in lines if line.strip())
    if not texts:
        print("Inference requires --text and/or --text-file.", file=sys.stderr)
        sys.exit(2)

    gold_labels: list[int] | None = None
    if ns.labels_file:
        gold_labels = load_gold_labels(ns.labels_file)
        if len(gold_labels) != len(texts):
            print(
                f"Label count ({len(gold_labels)}) must match text count ({len(texts)}).",
                file=sys.stderr,
            )
            sys.exit(2)

    device = pick_device(ns.force_cpu)
    ckpt = resolve_checkpoint_dir(ns.checkpoint, ns.output_dir)
    max_len = ns.max_length if ns.max_length is not None else 128

    print(f"Checkpoint: {ckpt}")
    print(f"Device: {device}\n")

    predictions = predict_sentences(texts, checkpoint_dir=ckpt, device=device, max_length=max_len)

    show_all = len(predictions) <= 10
    for i, row in enumerate(predictions):
        if gold_labels is not None:
            row["gold_label"] = ID2LABEL[gold_labels[i]]
            row["gold_label_id"] = gold_labels[i]
            row["correct"] = row["label_id"] == gold_labels[i]
        if show_all:
            print(f"Text: {row['text']!r}")
            print(f"Sentiment:  {row['label']} (confidence {row['confidence']:.4f})")
            print(f"Probs:      neg={row['prob_negative']:.4f}  pos={row['prob_positive']:.4f}")
            if gold_labels is not None:
                mark = "OK" if row["correct"] else "WRONG"
                print(f"Gold:       {row['gold_label']}  [{mark}]")
            print()

    if gold_labels is not None:
        correct = sum(1 for p, g in zip(predictions, gold_labels) if p["label_id"] == g)
        incorrect = len(predictions) - correct
        accuracy = correct / len(predictions)
        print("--- Evaluation ---")
        print(f"Correct:   {correct}/{len(predictions)} ({accuracy * 100:.1f}%)")
        print(f"Incorrect: {incorrect}/{len(predictions)} ({(1 - accuracy) * 100:.1f}%)")

        mistakes = [row for row in predictions if not row["correct"]]
        if mistakes:
            print(f"\nFirst {min(5, len(mistakes))} mistakes:")
            for row in mistakes[:5]:
                print(f"  gold={row['gold_label']} pred={row['label']} - {row['text'][:70]!r}")

        plot_path = ns.accuracy_plot
        if plot_path is None:
            plot_path = ns.plots_dir / "inference_accuracy.png"
        plot_inference_accuracy(
            correct=correct,
            incorrect=incorrect,
            accuracy=accuracy,
            out_path=plot_path,
        )
        print(f"\nSaved accuracy plot: {plot_path}")

        metrics_path = ns.plots_dir / "inference_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "total": len(predictions),
                    "correct": correct,
                    "incorrect": incorrect,
                    "accuracy": accuracy,
                    "accuracy_percent": round(accuracy * 100, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved metrics: {metrics_path}")
    elif not show_all:
        print(f"Classified {len(predictions)} sentences (use --labels-file to evaluate accuracy).")

    if ns.json_out:
        ns.json_out.parent.mkdir(parents=True, exist_ok=True)
        ns.json_out.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        print(f"Wrote {ns.json_out}")


def extract_eval_f1(results: dict) -> float:
    for key in ("eval_f1", "f1"):
        if key in results:
            return float(results[key])
    raise KeyError(f"No F1 in results: {list(results.keys())}")


def run_single_training(ns: argparse.Namespace, cfg: ResolvedRun, log: logging.Logger) -> dict:
    device = pick_device(ns.force_cpu)
    log.info("Device: %s | fp16=%s | attempt=%s | seed=%s", device, cfg.use_fp16, cfg.attempt, cfg.seed)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    dataset: DatasetDict = load_dataset(cfg.dataset)
    log.info("Dataset %s — train=%s validation=%s", cfg.dataset, len(dataset["train"]), len(dataset["validation"]))

    tokenize = partial(tokenize_sst2_batch, tokenizer=tokenizer, max_length=cfg.max_length)
    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_split = tokenized["train"]
    train_cap = cfg.max_train_samples
    if train_cap is None:
        train_cap = resolve_max_train_samples(ns, full_train_size=len(train_split))
    if train_cap is not None:
        n = min(train_cap, len(train_split))
        train_split = train_split.shuffle(seed=cfg.shuffle_seed).select(range(n))
        pct = 100.0 * n / len(tokenized["train"])
        log.info("Training on %s / %s rows (%.1f%% of train split)", n, len(tokenized["train"]), pct)

    training_args = build_training_arguments(cfg)
    callbacks = []
    if cfg.eval_strategy == "epoch" and cfg.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_split,
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    log.info(
        "Training: epochs=%s eval=%s lr=%s batch=%s/%s",
        cfg.epochs,
        cfg.eval_strategy,
        cfg.learning_rate,
        cfg.per_device_train_batch_size,
        cfg.per_device_eval_batch_size,
    )
    trainer.train()

    results = trainer.evaluate()
    eval_f1 = extract_eval_f1(results)
    log.info("Validation — F1=%.4f accuracy=%.4f loss=%.4f", eval_f1, results.get("eval_accuracy", 0), results.get("eval_loss", 0))

    serializable = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in results.items()}
    serializable["attempt"] = cfg.attempt
    serializable["seed"] = cfg.seed
    serializable["device"] = str(device)

    metrics_path = cfg.run_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    plot_path = cfg.plots_dir / "loss_curves.png"
    plot_training_curves(trainer.state.log_history, plot_path)
    shutil.copy2(plot_path, cfg.run_dir / "loss_curves.png")

    history_path = cfg.run_dir / "log_history.json"
    history_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    print_example_predictions(
        trainer,
        dataset["validation"],
        tokenizer,
        max_length=cfg.max_length,
        num_samples=cfg.num_example_predictions,
    )

    return {"results": results, "eval_f1": eval_f1, "trainer": trainer}


def main() -> None:
    ns = build_parser().parse_args()
    base = script_dir()
    runs_root = base / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    master_log = setup_file_logger(base / "logs" / "training_master.log")
    summary: list[dict] = []
    best_f1 = -1.0
    best_attempt = -1

    max_attempts = 1 if ns.no_auto_retry or ns.quick else ns.max_retries

    for attempt in range(max_attempts):
        cfg = resolve_run(ns, attempt=attempt, base_seed=ns.seed)
        attempt_log = setup_file_logger(cfg.run_dir / "train.log")

        attempt_log.info("========== Attempt %s / %s ==========", attempt + 1, max_attempts)
        if ns.quick:
            attempt_log.warning("--quick is for smoke tests only; use full training for report-quality metrics.")

        try:
            out = run_single_training(ns, cfg, attempt_log)
        except Exception:
            attempt_log.exception("Training failed on attempt %s", attempt + 1)
            summary.append({"attempt": attempt, "status": "failed"})
            if attempt + 1 >= max_attempts:
                raise
            shutil.rmtree(cfg.output_dir, ignore_errors=True)
            continue

        eval_f1 = out["eval_f1"]
        summary.append(
            {
                "attempt": attempt,
                "seed": cfg.seed,
                "eval_f1": eval_f1,
                "eval_accuracy": float(out["results"].get("eval_accuracy", 0)),
                "status": "ok",
            }
        )

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_attempt = attempt
            cfg.plots_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cfg.run_dir / "eval_metrics.json", cfg.plots_dir / "eval_metrics.json")

        if eval_f1 >= cfg.min_eval_f1:
            attempt_log.info("Target reached: eval_f1=%.4f >= %.4f", eval_f1, cfg.min_eval_f1)
            master_log.info("SUCCESS attempt %s — eval_f1=%.4f", attempt, eval_f1)
            break

        attempt_log.warning(
            "Below target: eval_f1=%.4f < %.4f — %s",
            eval_f1,
            cfg.min_eval_f1,
            "retrying" if attempt + 1 < max_attempts else "no more retries",
        )
        shutil.rmtree(cfg.output_dir, ignore_errors=True)

    summary_path = runs_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "best_attempt": best_attempt,
                "best_eval_f1": best_f1,
                "min_eval_f1": ns.min_eval_f1,
                "attempts": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    master_log.info("Done. Best eval_f1=%.4f (attempt %s). Summary: %s", best_f1, best_attempt, summary_path)

    if best_f1 < ns.min_eval_f1 and not ns.quick:
        master_log.error("Did not reach min_eval_f1=%.2f after %s attempts.", ns.min_eval_f1, max_attempts)
        sys.exit(1)


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.predict:
        run_inference(cli_args)
    else:
        main()
