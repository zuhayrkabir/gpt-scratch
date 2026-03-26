import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import os

# ── Argument parser ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='GPT Language Model Training')
parser.add_argument('--batch_size',     type=int,   default=128)
parser.add_argument('--block_size',     type=int,   default=64)
parser.add_argument('--max_iters',      type=int,   default=5000)
parser.add_argument('--learning_rate',  type=float, default=3e-4)
parser.add_argument('--eval_iters',     type=int,   default=100)
parser.add_argument('--n_embd',         type=int,   default=384)
parser.add_argument('--n_head',         type=int,   default=8)
parser.add_argument('--n_layer',        type=int,   default=8)
parser.add_argument('--dropout',        type=float, default=0.2)
parser.add_argument('--train_file',     type=str,   default='output_train.txt')
parser.add_argument('--val_file',       type=str,   default='output_val.txt')
parser.add_argument('--vocab_file',     type=str,   default='vocab.txt')
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
print(f"Using device: {device}")
print(f"batch_size={batch_size} | block_size={block_size} | max_iters={max_iters}")
print(f"n_embd={n_embd} | n_head={n_head} | n_layer={n_layer}")

# ── Vocab ──────────────────────────────────────────────────────────────────────
with open(args.vocab_file, 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))

vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s if c in string_to_int]
decode = lambda l: ''.join([int_to_string[i] for i in l])



def get_random_chunk(split):
    filename = "output_train.txt" if split == 'train' else "output_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B, T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x



class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)  

        logits = self.lm_head(x)
        

        if targets is None: 
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss


    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)  # ← index_cond here
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index


model = GPTLanguageModel(vocab_size)
m = model.to(device)

if os.path.exists('model-01.pkl'):
    print("loading model params...")
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
    m = model.to(device)
    print("loaded successfully")
else:
    print("no saved model found, starting fresh")



best_val_loss = float('inf')  # initialize to infinity


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, validation loss: {losses['val']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())




# Always save latest
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print("model saved!")

# Also save best if val loss improved
if losses['val'] < best_val_loss:
    best_val_loss = losses['val']
    with open('model-best.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("new best model saved!")




# ── Save final model ───────────────────────────────────────────────────────────
print(f"\nFinal loss: {loss.item():.4f}")
with open(args.save_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {args.save_path}")

# ── Quick generation test ──────────────────────────────────────────────────────
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(m.generate(context, max_new_tokens=200)[0].tolist())
print("\n--- Generated text ---")
print(generated)