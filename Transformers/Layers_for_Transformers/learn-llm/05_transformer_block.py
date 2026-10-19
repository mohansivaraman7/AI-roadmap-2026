"""
STEP 5 — THE TRANSFORMER BLOCK: the Lego brick of every LLM.

One block =
    x = x + MultiHeadAttention(LayerNorm(x))   # words talk to each other
    x = x + FeedForward(LayerNorm(x))          # each word "thinks" alone
Then stack N blocks (GPT-2 small: 12, big models: ~100).

Three helpers you'll see everywhere:
  * Residual (x + ...)  : keeps the original info, makes deep stacks trainable
  * LayerNorm           : keeps numbers in a healthy range
  * Positional encoding : attention has NO sense of order by itself!
                          We add a position signal to each word vector.

(We use PyTorch's built-in nn.MultiheadAttention = what you built in step 4.)

Run:  python 05_transformer_block.py
"""
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
D, HEADS = 16, 4

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]   # talk
        x = x + self.ffn(self.ln2(x))                       # think
        return x

def positions(T, d=D):
    """Classic sine/cosine positional encoding from 'Attention Is All You Need'."""
    pos = torch.arange(T).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe

model = nn.Sequential(Block(), Block(), Block()).eval()     # a 3-layer Transformer
print("Parameters in 3 blocks:", sum(p.numel() for p in model.parameters()))

vocab = {"dog": 0, "bites": 1, "man": 2}
emb = nn.Embedding(3, D)

def encode(sentence, use_positions):
    x = emb(torch.tensor([vocab[w] for w in sentence.split()]))
    if use_positions:
        x = x + positions(len(x))
    return model(x.unsqueeze(0))[0]

with torch.no_grad():
    for use_pos in (False, True):
        a = encode("dog bites man", use_pos)
        b = encode("man bites dog", use_pos)
        same = torch.allclose(a[0], b[2], atol=1e-5)   # vector for 'dog' in both
        print(f"\npositions={use_pos}: is 'dog' encoded the same in both sentences? {same}")
    print("\n -> Without positions, 'dog bites man' == 'man bites dog' to the model!")

print("\nWe now have every ingredient. Step 6: train a tiny GPT that writes text.")
