import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import os
import math
import matplotlib.pyplot as plt

# ── Argument parser ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='GPT Language Model Training')
parser.add_argument('--batch_size',     type=int,   default=64)
parser.add_argument('--block_size',     type=int,   default=128)
parser.add_argument('--max_iters',      type=int,   default=20000)
parser.add_argument('--learning_rate',  type=float, default=3e-4)
parser.add_argument('--eval_iters',     type=int,   default=100)
parser.add_argument('--n_embd',         type=int,   default=384)
parser.add_argument('--n_head',         type=int,   default=8)
parser.add_argument('--n_layer',        type=int,   default=8)
parser.add_argument('--dropout',        type=float, default=0.2)
parser.add_argument('--train_file',     type=str,   default='output_train.txt')
parser.add_argument('--val_file',       type=str,   default='output_val.txt')
parser.add_argument('--save_path',      type=str,   default='model-01.pkl')
args = parser.parse_args()

# ── Hyperparameters ────────────────────────────────────────────────────────────
batch_size    = args.batch_size
block_size    = args.block_size
max_iters     = args.max_iters
learning_rate = args.learning_rate
eval_iters    = args.eval_iters
n_embd        = args.n_embd
n_head        = args.n_head
n_layer       = args.n_layer
dropout       = args.dropout

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

print(f"Device: {device}")
print(f"batch_size={batch_size} | block_size={block_size} | max_iters={max_iters}")
print(f"n_embd={n_embd} | n_head={n_head} | n_layer={n_layer}")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
import tiktoken
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
decode = lambda l: enc.decode(l)
print(f"Vocab size: {vocab_size}")

# ── Data loading ───────────────────────────────────────────────────────────────
def get_random_chunk(split):
    filename = args.train_file if split == 'train' else args.val_file
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            while True:
                start_pos = random.randint(0, file_size - block_size * batch_size)
                mm.seek(start_pos)
                block = mm.read(block_size * batch_size - 1)
                decoded_block = block.decode('utf-8', errors='ignore')
                data = torch.tensor(encode(decoded_block), dtype=torch.long)
                if len(data) > block_size + batch_size:
                    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ── Model ──────────────────────────────────────────────────────────────────────
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
        self.token_embedding_table.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

# ── Load or create model ───────────────────────────────────────────────────────
model = GPTLanguageModel(vocab_size)
if os.path.exists('model-best.pkl'):
    print('Loading best model...')
    with open('model-best.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Loaded successfully - continuing training')
elif os.path.exists(args.save_path):
    print(f'Loading model from {args.save_path}...')
    with open(args.save_path, 'rb') as f:
        model = pickle.load(f)
    print('Loaded successfully - continuing training')
else:
    print('No saved model found - starting fresh')

m = model.to(device)
print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')

# ── Estimate loss ──────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda'):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ── Training loop ──────────────────────────────────────────────────────────────
optimizer     = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler        = torch.amp.GradScaler('cuda')
best_val_loss = float('inf')
train_losses  = []
val_losses    = []
loss_steps    = []

def get_lr(it):
    warmup_iters = 100
    min_lr = learning_rate / 10
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    decay = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * decay))

for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'step {iter:5d} | train: {losses["train"]:.4f} | val: {losses["val"]:.4f} | lr: {lr:.6f}')
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        loss_steps.append(iter)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            with open('model-best.pkl', 'wb') as f:
                pickle.dump(model, f)
            print(f'  -> new best saved (val: {best_val_loss:.4f})')

    xb, yb = get_batch('train')
    with torch.amp.autocast('cuda'):
        logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

print(f'\nFinal loss: {loss.item():.4f}')
with open(args.save_path, 'wb') as f:
    pickle.dump(model, f)
print(f'Model saved to {args.save_path}')

# ── Plot loss curve ────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(loss_steps, train_losses, label='Train loss')
plt.plot(loss_steps, val_losses,   label='Val loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
print('Loss curve saved to loss_curve.png')

# ── Quick generation test ──────────────────────────────────────────────────────
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(m.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)[0].tolist())
print("\n--- Generated text ---")
print(generated)