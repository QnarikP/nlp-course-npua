"""
SentencePiece BPE - CC-100 Armenian (Large Corpus & Vocabulary)

This script trains a BPE tokenizer on the CC-100 Armenian dataset (Wenzek et al., LREC 2020)
with a larger corpus (>=50k sentences) and larger vocabulary (8000), then compares tokenization
with the small model from Part 1.

Steps:
1. Load CC-100 Armenian and extract >=50,000 sentences to .txt
2. Train SentencePiece BPE with vocab_size=8000, character_coverage=1.0
3. Encode and decode 5 sentences
4. Compare average tokens per sentence: small model (Part 1) vs large model
5. Discussion: how larger corpus and vocabulary affect tokenization quality
"""

import io
import lzma
import os
import re
import urllib.request

import sentencepiece as spm

# ──────────────────────────────────────────────
# 1. Load CC-100 Armenian and extract sentences
# ──────────────────────────────────────────────

CC100_URL = "https://data.statmt.org/cc-100/hy.txt.xz"
RAW_FILE = "hy.txt.xz"

if not os.path.exists(RAW_FILE):
    print(f"Downloading {CC100_URL} ...")
    urllib.request.urlretrieve(CC100_URL, RAW_FILE)
    print(f"Saved to '{RAW_FILE}'")
else:
    print(f"Using cached '{RAW_FILE}'")

SENTENCE_END = re.compile(r"(?<=[։.!?])\s+|\n+")
MIN_SENTENCES = 50_000
CORPUS_TXT = "corpus_cc100_50k.txt"


def extract_sentences_from_xz(path, min_count):
    sentences = []
    with lzma.open(path, mode="rt", encoding="utf-8") as f:
        for line in f:
            for part in SENTENCE_END.split(line):
                s = part.strip()
                if len(s) > 1:
                    sentences.append(s)
                    if len(sentences) >= min_count:
                        return sentences
    return sentences


sentences = extract_sentences_from_xz(RAW_FILE, MIN_SENTENCES)
print(f"Extracted {len(sentences):,} sentences")

with open(CORPUS_TXT, "w", encoding="utf-8") as f:
    for s in sentences:
        f.write(s + "\n")
print(f"Saved to '{CORPUS_TXT}'")

# ──────────────────────────────────────────────
# 2. Train SentencePiece BPE (large model)
# ──────────────────────────────────────────────

MODEL_PREFIX_LARGE = "models/hy_bpe_cc100_8k"
VOCAB_SIZE = 8000

os.makedirs("models", exist_ok=True)
spm.SentencePieceTrainer.train(
    input=CORPUS_TXT,
    model_prefix=MODEL_PREFIX_LARGE,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,
)
print(f"Model saved: {MODEL_PREFIX_LARGE}.model and .vocab")

sp_large = spm.SentencePieceProcessor()
sp_large.load(f"{MODEL_PREFIX_LARGE}.model")
print(f"Loaded large model \u2014 vocab size: {sp_large.get_piece_size()}")

# ──────────────────────────────────────────────
# 3. Encode and decode 5 sentences
# ──────────────────────────────────────────────

TEST_SENTENCES = [
    "\u0540\u0561\u0575\u0561\u057d\u057f\u0561\u0576\u0576 \u0578\u0582\u0576\u056b \u0570\u0561\u0580\u0578\u0582\u057d\u057f \u057a\u0561\u057f\u0574\u0578\u0582\u0569\u0575\u0578\u0582\u0576\u0589",
    "\u0531\u0580\u0570\u0565\u057d\u057f\u0561\u056f\u0561\u0576 \u0562\u0561\u0576\u0561\u056f\u0561\u0576\u0578\u0582\u0569\u0575\u0578\u0582\u0576\u0568 \u0561\u0580\u0561\u0563 \u0566\u0561\u0580\u0563\u0561\u0576\u0578\u0582\u0574 \u0567\u0589",
    "\u053e\u0580\u0561\u0563\u0580\u0561\u057e\u0578\u0580\u0578\u0582\u0574\u0568 \u056f\u0561\u0580\u0587\u0578\u0580 \u0570\u0574\u057f\u0578\u0582\u0569\u0575\u0578\u0582\u0576 \u0567 \u0561\u057a\u0561\u0563\u0561\u0575\u056b \u0570\u0561\u0574\u0561\u0580\u0589",
    "\u0532\u0576\u0561\u056f\u0561\u0576 \u056c\u0565\u0566\u057e\u056b \u0574\u0577\u0561\u056f\u0578\u0582\u0574\u0568 \u0570\u0565\u057f\u0561\u0584\u0580\u0584\u056b\u0580 \u0563\u056b\u057f\u0578\u0582\u0569\u0575\u0561\u0576 \u0573\u0575\u0578\u0582\u0572 \u0567\u0589",
    "\u0544\u0565\u0584\u0565\u0576\u0561\u056f\u0561\u0576 \u0578\u0582\u057d\u0578\u0582\u0581\u0578\u0582\u0574\u0568 \u0583\u0578\u056d\u0578\u0582\u0574 \u0567 \u057f\u0565\u056d\u0576\u0578\u056c\u0578\u0563\u056b\u0561\u0576\u0589",
]

for i, sent in enumerate(TEST_SENTENCES, 1):
    pieces = sp_large.encode(sent, out_type=str)
    ids = sp_large.encode(sent, out_type=int)
    decoded = sp_large.decode(ids)
    match = "\u2713" if decoded == sent else "\u2717"
    print(f"S{i}: {sent}")
    print(f"  Pieces ({len(pieces)} tokens): {pieces}")
    print(f"  Decoded: {decoded}  {match}\n")

# ──────────────────────────────────────────────
# 4. Compare: small model (Part 1) vs large model
# ──────────────────────────────────────────────

SMALL_MODEL_PATH = "models/hy_bpe.model"
VOCAB_SIZE_SMALL = 600

if os.path.exists(SMALL_MODEL_PATH):
    sp_small = spm.SentencePieceProcessor()
    sp_small.load(SMALL_MODEL_PATH)
    print("Loaded Part 1 small model from", SMALL_MODEL_PATH)
else:
    MODEL_PREFIX_SMALL = "models/hy_bpe_small_600"
    spm.SentencePieceTrainer.train(
        input=CORPUS_TXT,
        model_prefix=MODEL_PREFIX_SMALL,
        vocab_size=VOCAB_SIZE_SMALL,
        model_type="bpe",
        character_coverage=1.0,
    )
    sp_small = spm.SentencePieceProcessor()
    sp_small.load(f"{MODEL_PREFIX_SMALL}.model")
    print("Trained small model (vocab 600) on same corpus for comparison")
print(f"Small model vocab size: {sp_small.get_piece_size()}\n")


def avg_tokens_per_sentence(sp, sents):
    counts = [len(sp.encode(s, out_type=int)) for s in sents]
    return sum(counts) / len(counts), counts


avg_small, counts_small = avg_tokens_per_sentence(sp_small, TEST_SENTENCES)
avg_large, counts_large = avg_tokens_per_sentence(sp_large, TEST_SENTENCES)

print("Tokens per sentence:")
print("  Small model (Part 1, vocab 600):", counts_small, "\u2192 avg =", round(avg_small, 2))
print("  Large model (CC-100, vocab 8000):", counts_large, "\u2192 avg =", round(avg_large, 2))
print("\nComparison:")
if avg_small:
    print(f"  Large model uses {avg_small - avg_large:.2f} fewer tokens per sentence on average")
    if avg_large > 0:
        print(f"  Small model uses {avg_small / avg_large:.2f}x more tokens than large model")

# ──────────────────────────────────────────────
# 5. Discussion
# ──────────────────────────────────────────────
# Larger corpus: With more text (50k+ sentences from CC-100), BPE learns more meaningful
# subword units. Frequent words and common morphemes are merged into single tokens instead
# of being split into many character-level pieces.
#
# Larger vocabulary: Increasing vocab_size (600 -> 8000) gives more slots for subword types.
# More whole words and common chunks become single tokens, so sentences are encoded with
# fewer, longer tokens.
#
# Combined effect: Both together typically reduce the average number of tokens per sentence.
# The tokenizer behaves more "word-like" and less "character-like," with better alignment
# to linguistic units. Round-trip (encode -> decode) remains lossless.
