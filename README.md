# GPT Language Model — From Scratch

A GPT-style transformer language model built from scratch in PyTorch, without using any pretrained models or high-level AI libraries. Every component of the architecture was implemented manually, pretrained on the OpenWebText corpus across multiple versions from a bigram baseline to a full transformer with BPE tokenization.

---

## What I Built

This is a decoder-only transformer architecture — the same design that powers GPT-2 and GPT-3 — implemented entirely from first principles. No `transformers` library, no pretrained weights. Just PyTorch and math.

The model learns to generate coherent English text by predicting the next token given the previous context, trained on millions of real web pages.

---

## Architecture

| Component | Details |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Attention | Multi-head causal self-attention |
| Layers | 8 transformer blocks |
| Attention heads | 8 |
| Embedding dimension | 384 |
| Context length | 128 tokens |
| Tokenization | Byte Pair Encoding (BPE) via tiktoken |
| Vocab size | 50,257 subword tokens |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Parameters | ~53M |

### Components built from scratch:
- **Multi-head causal self-attention** with scaled dot-product attention and triangular causal mask
- **Positional embeddings** — learned position vectors added to token embeddings
- **Residual connections** — `x = x + attention(x)` for stable gradient flow
- **Layer normalization** — Pre-LN style (normalize before attention, not after)
- **Feed-forward network** — 4x expansion with GELU activation
- **Weight initialization** — Normal distribution with std=0.02
- **Cosine learning rate schedule** with linear warmup
- **Gradient clipping** for training stability

---

## Training

- **Dataset:** OpenWebText — an open-source replication of the dataset used to train GPT-2, sourced from Reddit-linked web pages (~40GB total, ~400MB sampled)
- **Tokenization:** Byte Pair Encoding (BPE) via tiktoken — same tokenizer as GPT-2
- **Hardware:** NVIDIA RTX 5070 Laptop GPU (CUDA)
- **Optimizer:** AdamW (lr=3e-4, cosine decay)
- **Steps:** ~20,000
- **Final loss:** 4.77 (BPE) / 1.41 (character-level)

### Loss Curve
![Training Loss Curve](loss_curve.png)

Smooth, consistent downward trend over 20,000 steps from 6.0 → 4.7 with train and val loss tracking closely — no overfitting.

---

## Versions

| Version | Tokenization | Layers | Loss | Notes |
|---|---|---|---|---|
| v1 | Character-level | 1 | 2.34 | Bigram baseline |
| v2 | Character-level | 8 | 1.41 | Full transformer |
| v3 | BPE (tiktoken) | 8 | 4.77 | Word-level tokens, 20k steps |

> Note: v2 and v3 losses are not directly comparable — BPE has a vocabulary of 50,257 tokens vs ~17,000 characters, so random baseline loss is ~10.8 vs ~9.8. v3 represents significantly more learned structure per token.

---

## Sample Output

**v2 — Character-level (loss 1.41):**
```
The community of Fadder Brover in expeating Change Canada, big as 
best-two-challenge of campaign partically breakly, letter their political 
appletations five. Scorpor lices asked amongment party.
```

**v3 — BPE (loss 4.77, ~20k steps):**
```
Over the years the Orasard, who briefly pledged that conspiracy to support 
the meaning, gave a theological eyes, saying "aren't after anything, 
[the destruction of business liberty] to defend Jeremy signing in fighting 
the truth about it."

During more than a warning agreement, it follows that the Senate was taking 
the process to the enthusiastic US elections and the United States vote. 
The major change has not skipped these fantastic campaign may be reminiscent 
of the polls across small nations, throughout the country.

Taylor John DeMach insisted he had more in stock and darker shares than 
any other population.
```

v3 produces almost entirely real English words with correct sentence structure, real proper nouns (Senate, US elections, First Amendment), news article formatting and complex clause structure.

---

## Project Structure

```
gpt-from-scratch/
├── gpt-v2.ipynb        # Character-level transformer
├── gpt-v3.ipynb        # BPE transformer with training improvements
├── training.py         # Standalone training script with argparse
├── chatbot.py          # Interactive text generation interface
├── data-extract.py     # OpenWebText data extraction and preprocessing
├── loss_curve.png      # Training loss visualization
└── vocab.txt           # Character vocabulary (v2)
```

---

## How to Run

### Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install datasets tqdm matplotlib tiktoken
```

### Extract training data
```bash
python data-extract.py
```
This downloads OpenWebText via Hugging Face and generates `output_train.txt` and `output_val.txt`.

### Train the model
```bash
python training.py --max_iters 20000 --n_layer 8 --n_head 8
```

### Generate text interactively
```bash
python chatbot.py
```

---

## Key Technical Decisions

**Why BPE over character-level tokenization?**
Character-level tokenization requires the model to learn spelling before it can learn meaning. BPE encodes common words as single tokens so the model operates at the word level from the start. A context window of 128 BPE tokens covers a full paragraph vs just a few words at character level.

**Why train from scratch instead of fine-tuning?**
The goal was to understand the full pretraining pipeline — how a model goes from random weights to generating coherent text purely through next-token prediction on raw text data.

**Why cosine learning rate scheduling?**
A flat learning rate causes the model to overshoot optima in later training. Cosine decay with linear warmup lets the model take large steps early when gradients are large, then settle into finer adjustments as training progresses — typically improving final loss by 5-10%.

**Why OpenWebText?**
It's the standard open-source replication of the dataset used to train GPT-2, making results more comparable to published work than a toy dataset.

---

## What I Learned

- How transformer self-attention works mechanically — Q, K, V projections, scaled dot-product, causal masking
- Why residual connections and layer normalization are essential for training deep networks
- The full pretraining pipeline from raw text to a trained language model
- The difference between character-level and subword tokenization and how it affects what the model can learn
- CUDA debugging — device-side asserts, embedding table mismatches, memory management
- Memory-mapped file I/O for training on datasets too large to fit in RAM
- Training stability techniques — gradient clipping, learning rate scheduling, model checkpointing

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 paper
- [OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/)
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — architectural reference