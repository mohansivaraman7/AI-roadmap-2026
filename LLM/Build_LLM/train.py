"""
train.py -- train the tiny GPT on a text file and save everything needed to serve it
====================================================================================
    python train.py                              # trains on corpus.txt, ~1-3 min on a laptop CPU
    python train.py --text my_docs.txt --steps 3000 --n_layer 6 --n_embd 192

Saves into ./checkpoint/:  model.pt (weights + config), tokenizer.json
These two small files are all you need to host the model.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoint"
END = "<|end|>"


def train_tokenizer(text: str, vocab_size: int) -> Tokenizer:
    """Byte-level BPE: the same idea GPT-2 uses. Learns common word pieces from YOUR text."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        text.split("\n"),
        trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[END],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def run_training(text, steps=1500, batch=32, block=64, n_layer=4, n_head=4, n_embd=128,
                 lr=3e-3, vocab=800, on_step=None, out_dir=CKPT):
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = train_tokenizer(text, vocab)
    # every line ends with an END token so the model learns when to stop
    ids = []
    for line in text.split("\n"):
        if line.strip():
            ids += tok.encode(line).ids + [tok.token_to_id(END)]
    data = torch.tensor(ids)
    n_val = max(len(data) // 10, block + 2)
    train_data, val_data = data[:-n_val], data[-n_val:]

    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), block_size=block, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, dropout=0.1)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)   # same AdamW as before
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    def get_batch(d):
        ix = torch.randint(len(d) - block - 1, (batch,))
        x = torch.stack([d[i:i + block] for i in ix])
        y = torch.stack([d[i + 1:i + block + 1] for i in ix])      # target = input shifted by one token
        return x.to(device), y.to(device)

    history = []
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val = sum(model(*get_batch(val_data))[1].item() for _ in range(5)) / 5
            history.append((step, loss.item(), val))
            if on_step:
                on_step(step, steps, loss.item(), val)

    out_dir.mkdir(exist_ok=True)
    torch.save({"config": cfg.to_dict(), "state_dict": model.cpu().state_dict()}, out_dir / "model.pt")
    tok.save(str(out_dir / "tokenizer.json"))
    return model.cpu(), tok, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=str(HERE / "corpus.txt"))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    a = p.parse_args()
    text = Path(a.text).read_text(encoding="utf-8")
    t0 = time.time()
    m, tok, hist = run_training(text, steps=a.steps, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd,
                                on_step=lambda s, n, tl, vl: print(f"step {s:5d}/{n}  train {tl:.3f}  val {vl:.3f}"))
    print(f"\n{m.num_params():,} parameters, trained in {time.time() - t0:.0f}s -> checkpoint/")
