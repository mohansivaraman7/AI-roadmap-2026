"""
app.py -- the Streamlit app people use. Loads checkpoint/ and generates text.
============================================================================
    streamlit run app.py

Tab "Use":   type the start of a sentence, get the model's continuation.
Tab "Train": (optional) retrain in the browser on your own text, then use it straight away.
If checkpoint/ is missing (first run on a fresh host), it trains automatically on corpus.txt.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from train import CKPT, END, HERE, run_training


@st.cache_resource(show_spinner="Loading model...")
def load(stamp: float):
    ck = torch.load(CKPT / "model.pt", map_location="cpu")
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), Tokenizer.from_file(str(CKPT / "tokenizer.json"))


def complete(model, tok, prompt, max_new, temperature, top_k):
    ids = tok.encode(prompt).ids or [tok.token_to_id(END)]
    out = model.generate(torch.tensor([ids]), max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k, stop_id=tok.token_to_id(END))
    return tok.decode(out[0].tolist()).replace(END, "").strip()


st.set_page_config(page_title="My own tiny LLM", page_icon="\U0001F9E0")
st.title("My own tiny LLM")

if not (CKPT / "model.pt").exists():
    with st.spinner("First start: training the model on corpus.txt (about a minute)..."):
        run_training((HERE / "corpus.txt").read_text(encoding="utf-8"))
    st.rerun()

model, tok = load((CKPT / "model.pt").stat().st_mtime)
use_tab, train_tab = st.tabs(["Use", "Train"])

with use_tab:
    st.caption(f"{model.num_params():,} parameters, trained from random weights. It continues text; it does not follow instructions.")
    prompt = st.text_input("Start a sentence", value="Employees receive")
    c1, c2, c3 = st.columns(3)
    max_new = c1.slider("Max new tokens", 8, 120, 40)
    temperature = c2.slider("Temperature", 0.1, 1.5, 0.7, 0.1, help="Low = safe and repetitive, high = creative and wrong")
    top_k = c3.slider("Top-k", 1, 50, 20)
    if st.button("Generate", type="primary"):
        st.write(complete(model, tok, prompt, max_new, temperature, top_k))

with train_tab:
    st.write("Retrain from scratch on your own text. Random weights every time.")
    up = st.file_uploader("Training text (.txt, one sentence or paragraph per line)", type=["txt"])
    text = up.getvalue().decode("utf-8") if up else (HERE / "corpus.txt").read_text(encoding="utf-8")
    st.caption(f"{len(text):,} characters")
    a, b, c = st.columns(3)
    steps = a.number_input("Steps", 100, 20000, 1500, 100)
    n_layer = b.number_input("Layers", 1, 12, 4)
    n_embd = c.select_slider("Width", [64, 128, 192, 256], value=128)
    if st.button("Train"):
        bar, chart, losses = st.progress(0.0), st.empty(), []
        def on_step(s, n, tl, vl):
            losses.append({"train loss": tl, "validation loss": vl})
            bar.progress(s / n)
            chart.line_chart(pd.DataFrame(losses))
        m, _, _ = run_training(text, steps=int(steps), n_layer=int(n_layer), n_embd=int(n_embd), on_step=on_step)
        st.success(f"Trained a {m.num_params():,} parameter model. Open the Use tab.")
        load.clear()
