"""
STEP 2 — ATTENTION: instead of one squeezed summary, LOOK BACK at every word.

Story: translating "I love cats" -> French.
  The encoder (an LSTM) produces one vector per word: e1, e2, e3.
  Old way : decoder only gets the LAST vector (bottleneck).
  Attention: when writing each French word, the decoder asks
             "which English word matters right now?" and mixes them.

The 3 lines of attention:
  scores  = query · each key        (how relevant is each word?)
  weights = softmax(scores)         (turn into percentages, sum = 1)
  context = Σ weights * values      (weighted mix of the words)

Run:  python 02_attention.py
"""
import torch

torch.set_printoptions(precision=2, sci_mode=False)

words = ["I", "love", "cats"]
# Pretend these are the encoder's vectors for each word (4 numbers each).
# We hand-made them so the meaning is easy to see:
#            [person, emotion, animal, extra]
enc = torch.tensor([[1.0, 0.0, 0.0, 0.2],    # I
                    [0.0, 1.0, 0.0, 0.1],    # love
                    [0.0, 0.0, 1.0, 0.3]])   # cats

def attention(query, keys, values):
    scores = keys @ query                  # dot product with every word
    weights = torch.softmax(scores, dim=0) # percentages
    context = weights @ values             # weighted average
    return weights, context

# The decoder is about to write each French word. Its "query" says what it needs.
decoder_steps = {
    "Je     (needs a person)":  torch.tensor([3.0, 0.0, 0.0, 0.0]),
    "aime   (needs an emotion)": torch.tensor([0.0, 3.0, 0.0, 0.0]),
    "les chats (needs animal)": torch.tensor([0.0, 0.0, 3.0, 0.0]),
}

for french, query in decoder_steps.items():
    w, ctx = attention(query, enc, enc)
    bars = "  ".join(f"{word}:{p:.0%}" for word, p in zip(words, w))
    print(f"writing {french:28s} -> looks at  {bars}")

print("\nWhat to notice:")
print(" * Each output word gets its OWN mix of input words (no bottleneck).")
print(" * Weights are learned in a real model; here we hand-picked the query.")
print(" * Query asks, Keys answer 'how relevant am I', Values are what gets mixed.")
print("\nBut the encoder is still an LSTM (slow, step by step).")
print("Next idea: let words attend to EACH OTHER directly -> self-attention, step 3.")
