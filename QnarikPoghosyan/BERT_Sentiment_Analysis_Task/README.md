# BERT Sentiment Analysis (SST-2)

Fine-tune **BERT-base-uncased** on the **SST-2** movie-review dataset. The model predicts **positive** or **negative** sentiment.

## Setup

Use **Python 3.12** with GPU support (`venv_gpu`):

```powershell
cd BERT_Sentiment_Analysis_Task
py -3.12 -m venv venv_gpu --system-site-packages
.\venv_gpu\Scripts\Activate.ps1
pip install transformers datasets scikit-learn matplotlib accelerate tensorboard tf-keras
```

Check GPU:

```powershell
.\venv_gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

## Train

**Recommended** — 10% of training data, ~1–2 hours on GPU:

```powershell
.\venv_gpu\Scripts\python.exe bert_sentiment_analysis.py --train-fraction 0.1 --no-auto-retry
```

**Full dataset** (slower, best metrics):

```powershell
.\venv_gpu\Scripts\python.exe bert_sentiment_analysis.py
```

**Quick smoke test** (not for final results):

```powershell
.\venv_gpu\Scripts\python.exe bert_sentiment_analysis.py --quick
```

More options: `python bert_sentiment_analysis.py --help`

## Outputs

| File | What it is |
|------|------------|
| [`plots/loss_curves.png`](plots/loss_curves.png) | Training / validation loss and metrics |
| `plots/eval_metrics.json` | Best validation accuracy and F1 |
| `results/checkpoint-*` | Saved model checkpoints |

## Inference

Classify new text with a trained checkpoint:

```powershell
# Single sentence
.\venv_gpu\Scripts\python.exe bert_sentiment_analysis.py --predict --text "This film was brilliant."

# Many sentences from a file
.\venv_gpu\Scripts\python.exe bert_sentiment_analysis.py --predict --text-file sentences.txt
```

**Evaluate accuracy** against a labels file (one label per line: `positive` / `negative`):

```powershell
.\venv_gpu\Scripts\python.exe bert_sentiment_analysis.py --predict --text-file sentences.txt --labels-file labels.txt
```

This writes [`plots/inference_accuracy.png`](plots/inference_accuracy.png) with the percentage of correct predictions.

## Files

- `bert_sentiment_analysis.py` — train, evaluate, and predict
- `sentences.txt` / `labels.txt` — 100 example sentences with gold labels
- `requirements.txt` — Python dependencies
