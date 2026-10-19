"""
STEP 1 — LSTM: a network that reads words ONE AT A TIME and keeps a "memory".

Big idea:
  An RNN/LSTM walks through a sequence left to right.
  At each step it updates two vectors:
     h = hidden state  (what it "says" right now)
     c = cell state    (long-term memory)
  Gates (numbers between 0 and 1) decide:
     forget gate f : how much old memory to keep
     input  gate i : how much new info to write
     output gate o : how much memory to show as h

Part A: one LSTM cell written by hand (so you SEE the gates).
Part B: train PyTorch's LSTM to remember the FIRST number in a sequence.

Run:  python 01_lstm.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------- Part A: LSTM cell by hand ----------------
def lstm_cell(x, h, c, W, U, b):
    gates = x @ W + h @ U + b              # one big matrix multiply
    i, f, g, o = gates.chunk(4, dim=-1)    # split into 4 parts
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)  # 0..1
    g = torch.tanh(g)                      # candidate new memory
    c = f * c + i * g                      # forget some, write some
    h = o * torch.tanh(c)                  # read out
    return h, c

H, D = 3, 2                                 # hidden size, input size
W, U, b = torch.randn(D, 4 * H), torch.randn(H, 4 * H), torch.zeros(4 * H)
h, c = torch.zeros(H), torch.zeros(H)
sentence = torch.randn(4, D)                # 4 "words", each a 2-number vector
print("Part A — walking through 4 words one by one:")
for t, x in enumerate(sentence):
    h, c = lstm_cell(x, h, c, W, U, b)
    print(f"  word {t}: h = {h.numpy().round(2)}")
print("  -> the final h is a summary of the WHOLE sentence\n")

# ---------------- Part B: can it remember? ----------------
# Task: sequence of 10 numbers (0..4). Answer = the FIRST number.
# To succeed, the LSTM must carry it in memory across 9 steps.
SEQ, VOCAB = 10, 5

def make_batch(n=64):
    x = torch.randint(0, VOCAB, (n, SEQ))
    return x, x[:, 0]                       # target = first token

class Rememberer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 16)      # number -> vector
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.head = nn.Linear(32, VOCAB)        # vector -> guess
    def forward(self, x):
        out, (h, c) = self.lstm(self.emb(x))
        return self.head(h[-1])                 # use the LAST hidden state

model = Rememberer()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
print("Part B — training LSTM to remember the first number:")
for step in range(301):
    x, y = make_batch()
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        acc = (model(x).argmax(-1) == y).float().mean()
        print(f"  step {step:3d}  loss {loss.item():.3f}  accuracy {acc:.0%}")

x, _ = make_batch(1)
print(f"\n  input {x[0].tolist()} -> model says first number is {model(x).argmax().item()}")
print("\nKEY LIMITATION: everything must squeeze through ONE vector h, step by step.")
print("Long sentences = info fades + slow (no parallelism). Attention fixes this -> step 2.")
