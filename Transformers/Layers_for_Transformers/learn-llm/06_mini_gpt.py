"""
STEP 6 — MINI-GPT: a real (tiny) language model trained from scratch.

This is the SAME recipe as ChatGPT / Claude, only microscopic:
  1. Tokenize text      (here: each character is a token)
  2. Embed tokens + add positions
  3. Stack Transformer blocks with a CAUSAL mask
  4. Predict the NEXT token. Loss = how surprised the model was.
  5. Generate: predict a token, append it, repeat.

Trains in ~1-2 minutes on a laptop CPU.
Run:  python 06_mini_gpt.py
"""
import torch
import torch.nn as nn

torch.manual_seed(42)

# ---------- 1. data + tokenizer ----------
text = """an lstm reads one word at a time and keeps a memory.
attention lets a model look back at every word.
self attention lets every word look at every other word.
many heads let the model look in many ways at once.
a transformer stacks attention and feed forward layers.
a language model predicts the next word, again and again.
""" * 20
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)
data = torch.tensor(encode(text))
V, CTX, D, HEADS, LAYERS = len(chars), 32, 64, 4, 2
print(f"vocab size {V}, training chars {len(data)}")

def get_batch(n=32):
    ix = torch.randint(len(data) - CTX - 1, (n,))
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])  # shifted by one = next token
    return x, y

# ---------- 2-3. the model ----------
class Block(nn.Module):                       # identical to step 5
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)            # what the token is
        self.pos = nn.Embedding(CTX, D)          # where it is (learned positions)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)              # scores for every possible next char
    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)   # no peeking ahead
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))             # (batch, T, vocab)

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8):
        for _ in range(n):
            logits = self(idx[:, -CTX:])[:, -1] / temperature   # last position only
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

model = MiniGPT()
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  (GPT-3 has 175,000,000,000)\n")

# ---------- 4. training ----------
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
prompt = torch.tensor([encode("a ")])
for step in range(1501):
    x, y = get_batch()
    loss = nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        sample = decode(model.generate(prompt, 60)[0].tolist()).replace("\n", " / ")
        print(f"step {step:4d} | loss {loss.item():.2f} | {sample}")

# ---------- 5. generation ----------
print("\nFinal sample:\n")
print(decode(model.generate(torch.tensor([encode("attention ")]), 200)[0].tolist()))
print("\nGibberish -> real words as loss drops. Scale data + size up = an LLM.")
print("Next: don't train your own — USE a big LLM as the brain of an agent (step 7).")
