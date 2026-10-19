"""
STEP 4 — MULTI-HEAD ATTENTION: several self-attentions side by side.

Why? One head can only learn one "kind" of relationship.
  head 1 might track grammar  (verb -> its subject)
  head 2 might track position (look at previous word)
  head 3 might track meaning  ("bank" -> "river" or "money")

How: split the d-dim vector into n_heads smaller pieces,
     run self-attention in each piece, glue results back, mix with one Linear.

This is exactly what's inside GPT / Claude / Llama (just bigger).

Run:  python 04_multi_head.py
"""
import math
import torch
import torch.nn as nn

torch.manual_seed(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=16, n_heads=4):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # Wq, Wk, Wv in one layer
        self.out = nn.Linear(d_model, d_model)       # mixes the heads together

    def forward(self, x, causal=True):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, D) -> (B, heads, T, dk): give each head its own slice
        split = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)   # (B, h, T, T)
        if causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        self.weights = torch.softmax(scores, dim=-1)            # saved to inspect
        y = self.weights @ v                                    # (B, h, T, dk)

        y = y.transpose(1, 2).reshape(B, T, D)                  # glue heads back
        return self.out(y)

mha = MultiHeadAttention(d_model=16, n_heads=4)
x = torch.randn(1, 5, 16)          # batch of 1 sentence, 5 words, 16 numbers each
y = mha(x)

print("input  shape:", tuple(x.shape), " (batch, words, features)")
print("output shape:", tuple(y.shape), " <- same shape, so we can stack layers")
print("weights shape:", tuple(mha.weights.shape), " (batch, heads, words, words)\n")

for h in range(4):
    last = mha.weights[0, h, -1].detach()      # what the LAST word looks at
    print(f"head {h}: last word attends to " +
          "  ".join(f"w{i}:{p:.2f}" for i, p in enumerate(last)))
print("\nEach head has a DIFFERENT pattern = different viewpoint on the sentence.")
print("Next: wrap this with a feed-forward net + residuals = Transformer block, step 5.")
