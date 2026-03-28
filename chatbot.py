import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os
import tiktoken

# ── Device ─────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')
print(f"Device: {device}")

# ── Hyperparameters — must match the trained model ─────────────────────────────
block_size    = 128
n_embd        = 384
n_head        = 8
n_layer       = 8
dropout       = 0.0  # no dropout at inference

# ── Tokenizer ──────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
decode = lambda l: enc.decode(l)
print(f"Vocab size: {vocab_size}")

# ── Model definition ───────────────────────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, index, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# ── Load model ─────────────────────────────────────────────────────────────────
model_path = 'model-best.pkl' if os.path.exists('model-best.pkl') else 'model-01.pkl'
print(f'Loading model from {model_path}...')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
m = model.to(device)
m.eval()
print('Model loaded!')
print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')

# ── Interactive generation loop ────────────────────────────────────────────────
print("\n--- GPT Text Completion ---")
print("Type a prompt and press Enter. Type 'quit' to exit.")
print("Commands: :temp=0.8 :topk=40 :tokens=200 to adjust settings\n")

temperature = 0.8
top_k       = 40
max_tokens  = 200

while True:
    prompt = input("Prompt:\n> ")

    if prompt.lower() == 'quit':
        break

    # Handle settings commands
    if prompt.startswith(':temp='):
        temperature = float(prompt.split('=')[1])
        print(f"Temperature set to {temperature}")
        continue
    if prompt.startswith(':topk='):
        top_k = int(prompt.split('=')[1])
        print(f"Top-k set to {top_k}")
        continue
    if prompt.startswith(':tokens='):
        max_tokens = int(prompt.split('=')[1])
        print(f"Max tokens set to {max_tokens}")
        continue

    if not prompt.strip():
        continue

    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        generated = decode(m.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)[0].tolist())
    print(f'\nCompletion:\n{generated}\n')