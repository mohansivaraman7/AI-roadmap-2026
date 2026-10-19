"""
STEP 3 — SELF-ATTENTION: every word looks at every other word, all at once.

Difference from step 2:
  Step 2: decoder (query) looks at encoder words (keys/values).
  Step 3: the SAME sentence makes queries, keys and values.
          Q = X·Wq   K = X·Wk   V = X·Wv     (three learned matrices)

Formula (memorise this one!):
     Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V

  √d     : keeps scores from getting huge when vectors are long
  mask   : for GPT-style models, a word may NOT look at future words

No loop over time -> the whole sentence is processed in parallel (fast on GPUs).

Run:  python 03_self_attention.py
"""
import math
import torch

torch.manual_seed(1)
torch.set_printoptions(precision=2, sci_mode=False)

words = ["the", "cat", "sat", "on", "mat"]
T, d = len(words), 8

X = torch.randn(T, d)            # one embedding vector per word (normally learned)
Wq, Wk, Wv = torch.randn(d, d), torch.randn(d, d), torch.randn(d, d)

def self_attention(X, causal=False):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                 # (T, d) each
    scores = Q @ K.T / math.sqrt(d)                  # (T, T): every word vs every word
    if causal:                                       # hide the future
        future = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)          # each ROW sums to 1
    return weights @ V, weights                      # new vector per word

def show(weights, title):
    print(title)
    print("          " + "".join(f"{w:>6}" for w in words))
    for w, row in zip(words, weights):
        print(f"  {w:>6}: " + "".join(f"{p:6.2f}" for p in row))
    print()

out, w = self_attention(X)
show(w, "Full self-attention (row = word asking, column = word looked at):")

out, w = self_attention(X, causal=True)
show(w, "Causal (GPT) self-attention — upper-right is 0, no peeking ahead:")

print("Shapes: input", tuple(X.shape), "-> output", tuple(out.shape))
print("Every word now carries context from the words it attended to.")
print("Weights here are random; training makes 'sat' attend to 'cat', etc.")
print("\nOne attention pattern is limiting. Next: many in parallel -> multi-head, step 4.")
