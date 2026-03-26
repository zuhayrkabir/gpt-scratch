# GPT Language Model — From Scratch

A GPT-style transformer language model built from scratch in PyTorch, without using any pretrained models or high-level AI libraries. Every component of the architecture was implemented manually, trained on the OpenWebText corpus.

---

## What I Built

This is a decoder-only transformer architecture — the same design that powers GPT-2 and GPT-3 — implemented entirely from first principles. No `transformers` library, no pretrained weights. Just PyTorch and math.

The model learns to generate coherent English text by predicting the next character given the previous context, trained on millions of real web pages.

---

## Architecture

| Component | Details |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Attention | Multi-head causal self-attention |
| Layers | 8 transformer blocks |
| Attention heads | 8 |
| Embedding dimension | 384 |
| Context length | 64 characters |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Parameters | ~85M |

### Components built from scratch:
- **Multi-head causal self-attention** with scaled dot-product attention and triangular causal mask
- **Positional embeddings** — learned position vectors added to token embeddings
- **Residual connections** — `x = x + attention(x)` for stable gradient flow
- **Layer normalization** — Pre-LN style (normalize before attention, not after)
- **Feed-forward network** — 4x expansion with GELU activation
- **Weight initialization** — Normal distribution with std=0.02

---

## Training

- **Dataset:** OpenWebText — an open-source replication of the dataset used to train GPT-2, sourced from Reddit-linked web pages (~40GB total, ~400MB sampled)
- **Tokenization:** Character-level
- **Hardware:** NVIDIA RTX 5070 Laptop GPU (CUDA)
- **Optimizer:** AdamW (lr=3e-4)
- **Final loss:** ~1.41 cross-entropy

### Loss Curve
![Training Loss Curve](loss_curve.png)

Consistent downward trend over 10,000+ steps with train and val loss tracking closely — no overfitting.

---

## Sample Output

**After training:**
```
The community of Fadder Brover in expeating Change Canada, big as 
best-two-challenge of campaign partically breakly, letter their political 
appletations five. Scorpor lices asked amongment party.

Used Phone Inkey? I'm known could not noticement plan in the middle way, 
and to start before yourself surrows over Neutralize payers button's 
surpretessed, making profiles now accoutable as most fungustrials said 
"If yingered that's fundation. When."
```

The model has learned English sentence structure, punctuation, paragraph formatting and news/web writing style from the training data.

---

## Project Structure

```
gpt-from-scratch/
├── gpt-v2.ipynb        # Main notebook — model definition and training
├── training.py         # Standalone training script with argparse
├── chatbot.py          # Interactive text generation interface
├── data-extract.py     # OpenWebText data extraction and preprocessing
├── loss_curve.png      # Training loss visualization
└── vocab.txt           # Character vocabulary
```

---

## How to Run

### Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install datasets tqdm matplotlib
```

### Extract training data
```bash
python data-extract.py
```
This downloads OpenWebText via Hugging Face and generates `output_train.txt` and `output_val.txt`.

### Train the model
```bash
python training.py --max_iters 10000 --n_layer 8 --n_head 8
```

### Generate text interactively
```bash
python chatbot.py
```

---

## Key Technical Decisions

**Why character-level tokenization?**
Simpler to implement and understand than BPE. The model learns to construct words character by character, demonstrating it has genuinely learned language structure rather than memorizing word-level patterns.

**Why train from scratch instead of fine-tuning?**
The goal was to understand the full pretraining pipeline — how a model goes from random weights to generating coherent text purely through next-token prediction on raw text data.

**Why OpenWebText?**
It's the standard open-source replication of the dataset used to train GPT-2, making results more comparable to published work than a toy dataset like Shakespeare.

---

## What I Learned

- How transformer self-attention works mechanically — Q, K, V projections, scaled dot-product, causal masking
- Why residual connections and layer normalization are essential for training deep networks
- The full pretraining pipeline from raw text to a trained language model
- CUDA debugging — device-side asserts, embedding table mismatches, memory management
- Memory-mapped file I/O for training on datasets too large to fit in RAM

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 paper
- [OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/)
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — architectural reference