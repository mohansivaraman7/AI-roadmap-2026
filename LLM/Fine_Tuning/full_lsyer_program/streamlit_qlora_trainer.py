"""
streamlit_qlora_trainer.py -- QLoRA fine-tuning app: data files -> JSONL -> train -> test
=========================================================================================
One Streamlit app, three tabs:

  1. DATA   Load files from different domains (CSV, JSON, JSONL, TXT), map their columns,
            and convert everything into ONE model-independent train.jsonl.
  2. TRAIN  QLoRA fine-tuning (base model loaded in 4-bit) with a trainer you choose:
              - Trainer     (transformers): you write tokenizing + the collator yourself
              - SFTTrainer  (TRL): hand it the data and a peft_config, it does the rest
  3. TEST   Ask a question. See the ORIGINAL model and the FINE-TUNED model side by side.

QLoRA = LoRA + a 4-bit base model (nf4, double quantization). Same adapter, much less memory.
The two QLoRA-only steps are marked "QLoRA STEP" in the code.

RUN:
    pip install streamlit pandas torch transformers peft datasets accelerate trl bitsandbytes
    streamlit run streamlit_qlora_trainer.py

HARDWARE: needs an NVIDIA GPU + bitsandbytes (not Mac, not CPU). Free Google Colab T4 works.
The app checks this and disables the Train button if it is missing.

Sample data lives in the data/ folder next to this file (4 domains, 4 formats).
"""

import gc
import io
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
ADAPTER_DIR = HERE / "adapters"
JSONL_PATH = HERE / "train.jsonl"

PROMPT_TEMPLATE = "### Question:\n{q}\n\n### Answer:\n"

MODEL_CHOICES = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
TARGET_CHOICES = {
    "all-linear  (attention + MLP; usual choice for QLoRA)": "all-linear",
    "q_proj, v_proj  (smallest adapter; Llama/Qwen/Mistral names)": ["q_proj", "v_proj"],
    "c_attn  (GPT-2 style models)": ["c_attn"],
}


# =============================================================================
# SECTION 1: DATA CONVERSION  (any file -> list of {instruction, response, domain})
# =============================================================================
# The JSONL we save is MODEL-INDEPENDENT: just instruction / response / domain.
# The model-specific text format (prompt template, end marker) is added later,
# at training time, so the same train.jsonl works for any model.

def read_source(name: str, raw: bytes):
    """Returns ("table", DataFrame) for csv/json/jsonl, or ("text", str) for txt."""
    suffix = Path(name).suffix.lower()
    text = raw.decode("utf-8-sig")
    if suffix == ".csv":
        return "table", pd.read_csv(io.StringIO(text))
    if suffix == ".jsonl":
        return "table", pd.read_json(io.StringIO(text), lines=True)
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):                       # {"items": [...]} -> the list inside
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        return "table", pd.DataFrame(data)
    if suffix == ".txt":
        return "text", text
    raise ValueError(f"Unsupported file type: {suffix}")


def table_to_records(df, domain, mode, q_col=None, a_col=None, text_col=None, label_col=None):
    records = []
    for _, row in df.iterrows():
        if mode == "question_answer":
            q, a = str(row[q_col]).strip(), str(row[a_col]).strip()
        else:  # classification rows become an instruction: "Classify ... : <text>" -> label
            q = f"Classify the sentiment of this review as positive or negative: {str(row[text_col]).strip()}"
            a = str(row[label_col]).strip()
        records.append({"instruction": q, "response": a, "domain": domain})
    return records


def text_to_records(text, domain):
    """Parses blocks like:   Q: question ...   A: answer ...   (blank line between pairs)"""
    pairs = re.findall(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\n\s*Q:|\Z)", text, flags=re.S)
    return [{"instruction": q.strip(), "response": a.strip(), "domain": domain} for q, a in pairs]


def clean_records(records):
    """Drop empty rows and exact duplicates."""
    seen, out = set(), []
    for r in records:
        key = (r["instruction"], r["response"])
        if r["instruction"] and r["response"] and r["instruction"] != "nan" and r["response"] != "nan" and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def save_jsonl(records, path=JSONL_PATH):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path=JSONL_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def guess_column(columns, keywords):
    for col in columns:
        if any(k in str(col).lower() for k in keywords):
            return col
    return columns[0]


# =============================================================================
# SECTION 2: TRAINING  (LoRA, with Trainer or SFTTrainer)
# =============================================================================

def get_device_info():
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return False, "CPU"


def qlora_ready():
    """QLoRA needs a CUDA GPU and bitsandbytes. Returns (ok, message)."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed."
    if not torch.cuda.is_available():
        return False, "No NVIDIA GPU found. QLoRA needs CUDA (bitsandbytes does not run on Mac or CPU). Use Google Colab (free T4) or use the LoRA app instead."
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        return False, f"bitsandbytes is not usable: {e}. Run: pip install bitsandbytes"
    return True, f"GPU ready: {torch.cuda.get_device_name(0)}"


def bnb_config():
    """QLoRA STEP 1: describe the 4-bit loading."""
    import torch
    from transformers import BitsAndBytesConfig
    use_bf16 = torch.cuda.is_bf16_supported()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",                    # 4-bit format designed for LLM weights
        bnb_4bit_use_double_quant=True,               # also compress the quantization constants
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )


def load_model_kwargs():
    """Use bf16 on GPUs that support it, else float32 (also right for CPU)."""
    import torch
    key = "dtype"
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        if (major, minor) < (4, 56):
            key = "torch_dtype"          # older transformers used this name
    except Exception:
        pass
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {key: torch.bfloat16}
    return {key: torch.float32}


def to_training_rows(records, eos_token):
    rows = []
    for r in records:
        prompt = PROMPT_TEMPLATE.format(q=r["instruction"])
        rows.append({"prompt": prompt, "completion": r["response"] + eos_token,
                     "answer": r["response"]})
    return rows


class PadCollator:
    """The 'collator': pads a batch to equal length. Labels use -100 on padding so the
    loss ignores it. (Real EOS tokens are kept, so the model learns when to stop.)"""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        longest = max(len(f["input_ids"]) for f in features)
        ids, labels, mask = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = longest - n
            ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            mask.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def make_progress_callback(progress_bar, chart_placeholder, status_placeholder):
    from transformers import TrainerCallback

    class LiveCallback(TrainerCallback):
        def __init__(self):
            self.losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.losses.append(logs["loss"])
                chart_placeholder.line_chart(pd.DataFrame({"training loss": self.losses}))
            if state.max_steps:
                done = min(state.global_step / state.max_steps, 1.0)
                progress_bar.progress(done)
                status_placeholder.caption(f"step {state.global_step} / {state.max_steps}"
                                           + (f"   loss {logs['loss']:.4f}" if logs and "loss" in logs else ""))

    return LiveCallback()


def train_lora(cfg, records, callback):
    """Trains a QLoRA adapter. Returns a small dict of results."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # QLoRA STEP 1: load the base model in 4-bit (the only change in the loading line)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb_config(), device_map={"": 0}, **load_model_kwargs())
    # QLoRA STEP 2: make the 4-bit model safe to train (casts norms to fp32, enables gradient checkpointing).
    # SFTTrainer does NOT do this for you, so we do it here for both trainers.
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
        target_modules=cfg["targets"], task_type="CAUSAL_LM",
    )

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    common = dict(
        output_dir=str(ADAPTER_DIR / "_runs"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["accum"],
        learning_rate=cfg["lr"],
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=bf16,
        fp16=use_cuda and not bf16,
        use_cpu=not use_cuda,
        optim="paged_adamw_8bit",                 # QLoRA's memory-saving optimizer (from bitsandbytes)
    )
    rows = to_training_rows(records, tokenizer.eos_token)
    answer_only = cfg["answer_only"]

    if cfg["mechanism"] == "Trainer":
        # ---- PLAIN Trainer: YOU do the tokenizing, the label masking and the collator ----
        model = get_peft_model(model, lora_config)          # YOU wrap the model with LoRA

        def tokenize(example):
            p_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
            ids = (p_ids + c_ids)[: cfg["max_len"]]
            labels = list(ids)
            if answer_only:                                   # ignore the question in the loss
                labels[: len(p_ids)] = [-100] * min(len(p_ids), len(labels))
            return {"input_ids": ids, "labels": labels}

        dataset = Dataset.from_list(rows).map(tokenize, remove_columns=["prompt", "completion", "answer"])
        trainer = Trainer(
            model=model, args=TrainingArguments(**common), train_dataset=dataset,
            data_collator=PadCollator(tokenizer.pad_token_id), callbacks=[callback],
        )
    else:
        # ---- SFTTrainer (TRL): give it data + peft_config; it tokenizes, masks and pads ----
        from trl import SFTConfig, SFTTrainer
        if answer_only:
            # prompt + completion columns -> TRL computes loss on the completion only
            dataset = Dataset.from_list([{"prompt": r["prompt"], "completion": r["answer"]} for r in rows])
            extra = {}
        else:
            dataset = Dataset.from_list([{"text": r["prompt"] + r["answer"]} for r in rows])
            extra = {"dataset_text_field": "text"}
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(max_length=cfg["max_len"], **extra, **common),
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,                          # TRL wraps the model with LoRA for you
            callbacks=[callback],                             # (TRL also adds the end marker itself)
        )

    trainable, total = trainer.model.get_nb_trainable_parameters()
    trainer.train()

    out_dir = ADAPTER_DIR / cfg["adapter_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)                  # saves only the small adapter
    tokenizer.save_pretrained(out_dir)
    (out_dir / "app_meta.json").write_text(json.dumps({"base_model": cfg["base_model"], "mode": "qlora",
                                                       "mechanism": cfg["mechanism"]}))
    final_loss = next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None)

    del trainer, model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return {"adapter_dir": str(out_dir), "trainable": trainable, "total": total, "final_loss": final_loss}


# =============================================================================
# SECTION 3: TESTING  (original model vs fine-tuned model)
# =============================================================================

@st.cache_resource(show_spinner="Loading base model and adapter...")
def load_for_test(adapter_dir: str, stamp: float, four_bit: bool):
    """`stamp` (file modified time) makes the cache refresh after re-training."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    meta = json.loads((Path(adapter_dir) / "app_meta.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if four_bit:
        base = AutoModelForCausalLM.from_pretrained(meta["base_model"], quantization_config=bnb_config(),
                                                    device_map={"": 0}, **load_model_kwargs())
    else:                                            # full-precision base: also works on CPU
        base = AutoModelForCausalLM.from_pretrained(meta["base_model"], **load_model_kwargs())
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question, use_adapter, max_new_tokens=60):
    import torch
    inputs = tokenizer(PROMPT_TEMPLATE.format(q=question), return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**inputs, **kwargs)
        else:
            with model.disable_adapter():                    # same model, adapter switched OFF = original
                out = model.generate(**inputs, **kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).split("###")[0].strip()


# =============================================================================
# SECTION 4: THE STREAMLIT UI
# =============================================================================

def tab_data():
    st.subheader("1. Load files and convert to JSONL")
    st.caption("Different domains, different formats. Each file is mapped to the same two fields: instruction and response.")

    use_samples = st.checkbox(f"Use the sample files in {DATA_DIR.name}/ (IT helpdesk, banking, hotel, product reviews)", value=True)
    uploads = st.file_uploader("...and/or upload your own files", type=["csv", "json", "jsonl", "txt"], accept_multiple_files=True)

    sources = []
    if use_samples and DATA_DIR.exists():
        sources += [(p.name, p.read_bytes()) for p in sorted(DATA_DIR.iterdir()) if p.suffix in {".csv", ".json", ".jsonl", ".txt"}]
    sources += [(u.name, u.getvalue()) for u in uploads]
    if not sources:
        st.info("Add at least one file.")
        return

    all_records = []
    for name, raw in sources:
        try:
            kind, content = read_source(name, raw)
        except Exception as e:
            st.error(f"{name}: {e}")
            continue
        with st.expander(f"{name}  ({kind})", expanded=False):
            domain = st.text_input("Domain tag", value=Path(name).stem, key=f"{name}-domain")
            if kind == "text":
                st.code(content[:400] + ("..." if len(content) > 400 else ""))
                recs = text_to_records(content, domain)
                st.caption(f"Found {len(recs)} Q:/A: pairs. (Plain text with no Q:/A: lines cannot become question-and-answer data without help.)")
            else:
                st.dataframe(content.head(5), width="stretch")
                cols = list(content.columns)
                mode = st.radio("How should this file be used?",
                                ["question_answer", "classification"],
                                format_func=lambda m: "Question and answer columns" if m == "question_answer" else "Text + label (classification turned into an instruction)",
                                key=f"{name}-mode", horizontal=True)
                if mode == "question_answer":
                    q_col = st.selectbox("Question column", cols, index=cols.index(guess_column(cols, ["question", "instruction", "prompt", "q"])), key=f"{name}-q")
                    a_col = st.selectbox("Answer column", cols, index=cols.index(guess_column(cols, ["answer", "response", "completion", "a"])), key=f"{name}-a")
                    recs = table_to_records(content, domain, mode, q_col=q_col, a_col=a_col)
                else:
                    t_col = st.selectbox("Text column", cols, index=cols.index(guess_column(cols, ["text", "review"])), key=f"{name}-t")
                    l_col = st.selectbox("Label column", cols, index=cols.index(guess_column(cols, ["label", "class", "sentiment"])), key=f"{name}-l")
                    recs = table_to_records(content, domain, mode, text_col=t_col, label_col=l_col)
            st.write(f"{len(recs)} examples from this file")
        all_records += recs

    if st.button("Convert to train.jsonl", type="primary"):
        records = clean_records(all_records)
        save_jsonl(records)
        st.session_state["records"] = records
        st.success(f"Saved {len(records)} examples to {JSONL_PATH.name}")

    records = st.session_state.get("records")
    if records:
        df = pd.DataFrame(records)
        st.write("Examples per domain")
        st.bar_chart(df["domain"].value_counts())
        st.write("First lines of train.jsonl")
        st.code("\n".join(json.dumps(r, ensure_ascii=False) for r in records[:3]), language="json")
        st.download_button("Download train.jsonl", JSONL_PATH.read_bytes() if JSONL_PATH.exists() else b"", file_name="train.jsonl")


def tab_train():
    st.subheader("2. Train a QLoRA adapter")
    if "records" not in st.session_state and JSONL_PATH.exists():
        st.session_state["records"] = load_jsonl()
    records = st.session_state.get("records")
    if not records:
        st.info("Go to the Data tab and convert your files first.")
        return
    st.write(f"Training data: {len(records)} examples from {len({r['domain'] for r in records})} domains")

    left, right = st.columns(2)
    with left:
        base_model = st.selectbox("Base model", MODEL_CHOICES)
        custom = st.text_input("...or any Hugging Face model id / local folder", value="")
        mechanism = st.radio("Trainer mechanism", ["SFTTrainer", "Trainer"], horizontal=True,
                             help="Trainer: you write tokenizing and the collator. SFTTrainer: TRL does it for you.")
        answer_only = st.checkbox("Learn only from the answer (ignore the question in the loss)", value=True)
        adapter_name = st.text_input("Adapter name", value=f"qlora-{mechanism.lower()}")
    with right:
        r = st.select_slider("LoRA rank r", options=[4, 8, 16, 32, 64], value=8)
        alpha = st.select_slider("lora_alpha", options=[8, 16, 32, 64, 128], value=16)
        dropout = st.slider("lora_dropout", 0.0, 0.3, 0.05, 0.05)
        target_label = st.selectbox("target_modules", list(TARGET_CHOICES))
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Epochs", 1, 50, 5)
    lr = c2.select_slider("Learning rate", options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4, format_func=lambda v: f"{v:g}")
    batch = c3.number_input("Batch size", 1, 32, 2)
    accum = c4.number_input("Grad accumulation", 1, 32, 1)
    max_len = st.number_input("Max length (tokens)", 32, 2048, 256)

    ready, msg = qlora_ready()
    (st.success if ready else st.error)(msg)

    if st.button("Start training", type="primary", disabled=not ready):
        cfg = dict(base_model=custom.strip() or base_model, mechanism=mechanism, answer_only=answer_only,
                   adapter_name=adapter_name.strip() or "qlora-adapter", r=r, alpha=alpha, dropout=dropout,
                   targets=TARGET_CHOICES[target_label], epochs=epochs, lr=lr, batch=batch, accum=accum, max_len=max_len)
        progress = st.progress(0.0)
        status = st.empty()
        chart = st.empty()
        callback = make_progress_callback(progress, chart, status)
        try:
            with st.spinner("Training..."):
                result = train_lora(cfg, records, callback)
        except Exception as e:
            st.error(f"Training failed: {e}")
            return
        progress.progress(1.0)
        st.session_state["adapter_dir"] = result["adapter_dir"]
        st.success(f"Done. Adapter saved to {result['adapter_dir']}")
        st.write(f"Trainable parameters: {result['trainable']:,} of {result['total']:,} "
                 f"({100 * result['trainable'] / result['total']:.3f}%)")
        if result["final_loss"] is not None:
            st.write(f"Final training loss: {result['final_loss']:.4f}")
        st.info("Open the Test tab to compare the original and fine-tuned model.")


def tab_test():
    st.subheader("3. Test: original model vs fine-tuned model")
    adapters = sorted(p for p in ADAPTER_DIR.glob("*") if (p / "app_meta.json").exists()) if ADAPTER_DIR.exists() else []
    if not adapters:
        st.info("Train an adapter first.")
        return
    names = [p.name for p in adapters]
    default = names.index(Path(st.session_state["adapter_dir"]).name) if st.session_state.get("adapter_dir") and Path(st.session_state["adapter_dir"]).name in names else 0
    chosen = adapters[names.index(st.selectbox("Adapter", names, index=default))]
    meta = json.loads((chosen / "app_meta.json").read_text())
    st.caption(f"Base model: {meta['base_model']}   |   trained with: {meta['mechanism']}")

    question = st.text_input("Ask a question from one of your domains", value="How long does a domestic transfer take?")
    max_new = st.slider("Max new tokens", 16, 200, 60)
    ok, _ = qlora_ready()
    four_bit = st.checkbox("Load the base model in 4-bit for testing", value=ok, disabled=not ok,
                           help="Unchecked = full precision (works on CPU, needs more memory).")
    if st.button("Ask", type="primary") and question.strip():
        model, tokenizer = load_for_test(str(chosen), (chosen / "adapter_model.safetensors").stat().st_mtime, four_bit)
        a, b = st.columns(2)
        with a:
            st.markdown("**Original model**")
            st.write(generate_answer(model, tokenizer, question, use_adapter=False, max_new_tokens=max_new))
        with b:
            st.markdown("**Fine-tuned model**")
            st.write(generate_answer(model, tokenizer, question, use_adapter=True, max_new_tokens=max_new))
    st.caption("Tip: also ask something that is NOT in your data, to see whether the model makes up an answer.")


def main():
    st.set_page_config(page_title="QLoRA fine-tuning", page_icon="\U0001F9EA", layout="wide")
    st.title("QLoRA fine-tuning: files \u2192 JSONL \u2192 train \u2192 test")
    t1, t2, t3 = st.tabs(["1. Data", "2. Train", "3. Test"])
    with t1:
        tab_data()
    with t2:
        tab_train()
    with t3:
        tab_test()


if __name__ == "__main__":
    main()
