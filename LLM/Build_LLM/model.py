"""
model.py -- a complete (tiny) GPT, written from scratch in PyTorch
===================================================================
The same design as GPT-2 / Llama in miniature:

   token ids -> embedding + position embedding
             -> N x [ LayerNorm -> causal self-attention (+ residual)
                      LayerNorm -> MLP                    (+ residual) ]
             -> LayerNorm -> linear head -> scores for the next token

Nothing here is pretrained. Weights start random and learn from your text.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1000
    block_size: int = 128     # context length: how many tokens the model sees at once
    n_layer: int = 4          # number of transformer blocks
    n_head: int = 4           # attention heads per block
    n_embd: int = 128         # width of every token vector
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)     # query, key, value in one matrix
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.n_head, self.drop = c.n_head, c.dropout
        self.resid_drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # is_causal=True: a token may only look at EARLIER tokens (that is what makes it a language model)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.n_embd), CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd), nn.Dropout(c.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # residual connections keep gradients healthy
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight               # weight tying: output layer shares the embedding
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, temperature=0.8, top_k=20, stop_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.c.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, -float("inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None and nxt.item() == stop_id:
                break
        return idx
