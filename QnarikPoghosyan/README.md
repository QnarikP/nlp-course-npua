# NLP Course

Homework and project assignments for the **Natural Language Processing** course at the **National Polytechnic University of Armenia (NPUA)**, Master's program, second semester.

Each directory contains a standalone project with its own dependencies, instructions, and documentation.

---

## Projects

| # | Project                                                                   | Description                                                              |
|---|---------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 1 | [LDA Topic Modeling](lda_topic_modeling/)                                 | Unsupervised topic discovery using Latent Dirichlet Allocation (Gensim)  |
| 2 | [SentencePiece BPE](sentencepiece_bpe/)                                   | BPE tokenizer trained on an Armenian corpus using SentencePiece          |
| 3 | [BERT Sentiment Analysis](BERT_Sentiment_Analysis_Task/)                  | Fine-tune BERT on SST-2 for binary movie-review sentiment classification  |

---

## Project Summaries

### 1. LDA Topic Modeling — [`lda_topic_modeling/`](lda_topic_modeling/)

An end-to-end pipeline for training, labeling, and running inference with an LDA topic model.

- Trains an LDA model (7 topics, 30 passes) on a text corpus
- Provides an interactive interface for assigning human-readable labels to discovered topics
- Classifies new documents and returns per-topic probability scores
- Generates visualizations: word clouds, heatmaps, bar charts, and document-topic distributions

**Stack:** Python, Gensim, NLTK, Matplotlib, WordCloud

---

### 2. SentencePiece BPE Tokenizer — [`sentencepiece_bpe/`](sentencepiece_bpe/)

Trains a BPE tokenizer on a small Armenian corpus and analyses the resulting vocabulary, then scales up to a large CC-100 corpus.

- Trains a 300-token BPE model with full Armenian Unicode character coverage
- Encodes and decodes three test sentences with round-trip verification
- Analyses vocabulary structure: single characters, subword fragments, and full words
- Generates four plots: vocab composition donut, token frequency chart, length histogram, and sentence tokenization diagram
- Trains a large BPE model (vocab 8000) on 50k+ sentences from CC-100 Armenian and compares tokenization with the small model

**Stack:** Python, SentencePiece, Matplotlib, NumPy, Hugging Face Datasets

---

### 3. BERT Sentiment Analysis — [`BERT_Sentiment_Analysis_Task/`](BERT_Sentiment_Analysis_Task/)

Fine-tunes **BERT-base-uncased** on the **SST-2** dataset to classify movie reviews as **positive** or **negative**.

- Trains and evaluates on SST-2 (full data or a faster 10% subset)
- Saves training curves and validation metrics under `plots/`
- Runs inference on new sentences via `--predict`
- Evaluates a custom 100-sentence test set (`sentences.txt` + `labels.txt`) and plots accuracy

**Stack:** Python, PyTorch, Hugging Face Transformers, Datasets, scikit-learn, Matplotlib

See [`BERT_Sentiment_Analysis_Task/README.md`](BERT_Sentiment_Analysis_Task/README.md) for setup and commands.

---

## Repository Structure

```text
QnarikPoghosyan/
├── lda_topic_modeling/              # Homework 1 — LDA Topic Modeling
├── sentencepiece_bpe/               # Homework 2 — SentencePiece BPE Tokenizer
└── BERT_Sentiment_Analysis_Task/   # Homework 3 — BERT SST-2 Sentiment
    ├── bert_sentiment_analysis.py
    ├── sentences.txt
    ├── labels.txt
    ├── plots/
    └── README.md
```

---

## Requirements

Each project manages its own virtual environment and dependencies. Refer to the `README.md` inside each project directory for setup instructions.

General prerequisites:

- Python 3.8+
- `pip`

---
