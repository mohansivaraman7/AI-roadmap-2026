"""
lora_finetune_full.py -- LoRA fine-tuning, complete: data -> train -> save -> reload -> answer
==============================================================================================
LoRA only (no quantization, no bitsandbytes), using the plain transformers Trainer.
Based on lora_finetune.py, with these additions:
  * works without policy_training_data.jsonl (uses built-in examples if the file is missing)
  * labels are built by hand, so the real end-of-text token is learned (the old
    DataCollatorForLanguageModeling hid it because pad token == eos token)
  * the question part of each example is ignored in the loss: the model learns the ANSWER
  * after training it RELOADS the saved adapter from disk, the way you would use it later
  * shows the ORIGINAL model and the FINE-TUNED model answering the same questions
  * interactive mode: ask your own questions

RUN:
    python lora_finetune_full.py                          # train, save, test
    python lora_finetune_full.py --ask                    # skip training, just chat with the saved adapter
    python lora_finetune_full.py --model Qwen/Qwen2.5-0.5B --epochs 10

SETUP:
    pip install torch transformers peft datasets accelerate
A GPU is recommended. A small model such as distilgpt2 also runs on CPU, slowly.

DATA FORMAT (policy_training_data.jsonl): one JSON object per line, either
    {"instruction": "question text", "response": "answer text"}
 or {"text": "### Question:\\n...\\n\\n### Answer:\\n...."}   (your earlier create_training_data.py format)
"""

import argparse
import json
import os

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BASE_MODEL = "distilgpt2"
DATA_FILE = "policy_training_data.jsonl"
ADAPTER_DIR = "policy_lora_adapter"
PROMPT_TEMPLATE = "### Question:\n{q}\n\n### Answer:\n"

BUILT_IN_EXAMPLES = [
    ("How many days of annual leave do employees get?", "All full-time employees receive 20 days of annual leave per calendar year."),
    ("How far in advance should annual leave be requested?", "Annual leave should normally be requested at least 7 days in advance."),
    ("Can unused annual leave be carried forward?", "Unused annual leave can be carried forward up to 5 days into the following year."),
    ("How many sick leave days do employees get per year?", "Employees receive 10 days of sick leave per year."),
    ("When is a medical certificate required for sick leave?", "If an employee is absent for more than 3 consecutive working days, a medical certificate may be required."),
    ("How many days per week can employees work from home?", "Employees can work from home 2 days per week, with manager approval."),
    ("What are the normal working hours?", "Normal working hours are 9:00 AM to 6:00 PM, Monday to Friday, with a 1-hour lunch break."),
    ("When are employees paid?", "Employees are paid on the last working day of each month."),
    ("How often do employees receive a performance review?", "Employees receive a formal performance review once every year."),
]

TEST_QUESTIONS = [
    "How many days of annual leave do employees get?",
    "When are employees paid?",
    "What is the company dress code?",      # NOT in the training data: watch what each model does
]


# ============================================================
# 1. DATA: read the JSONL (or fall back), return (question, answer) pairs
# ============================================================

def load_pairs(path=DATA_FILE):
    if not os.path.exists(path):
        print(f"{path} not found: using {len(BUILT_IN_EXAMPLES)} built-in examples.")
        return list(BUILT_IN_EXAMPLES)
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "instruction" in row and "response" in row:
                pairs.append((row["instruction"], row["response"]))
            elif "text" in row and "### Answer:" in row["text"]:
                q, a = row["text"].split("### Answer:", 1)
                pairs.append((q.replace("### Question:", "").strip(), a.strip()))
    print(f"Loaded {len(pairs)} examples from {path}")
    return pairs


# ============================================================
# 2. TOKENIZE: build input_ids and labels by hand
# ============================================================
# labels = the tokens the model is graded on. -100 means "ignore".
#   question part  -> -100  (we do not want to teach it to write questions)
#   answer + EOS   -> real ids (so it learns the answer AND when to stop)

def build_features(pairs, tokenizer, max_length=128):
    features = []
    for q, a in pairs:
        prompt_ids = tokenizer(PROMPT_TEMPLATE.format(q=q), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(a, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        ids = (prompt_ids + answer_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
        features.append({"input_ids": ids, "labels": labels})
    return features


class PadCollator:
    """Pads a batch to the same length. Padding is ignored by attention and by the loss."""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        longest = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = longest - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]


# ============================================================
# 3. MODEL + LoRA
# ============================================================

def default_targets(model):
    """Attention layer names differ by architecture."""
    return ["c_attn"] if model.config.model_type == "gpt2" else ["q_proj", "v_proj"]


def load_base(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32)
    return model, tokenizer


# ============================================================
# 4. TRAIN + SAVE
# ============================================================

def train(model_name, epochs, batch_size, lr, r, alpha, data_file, adapter_dir):
    model, tokenizer = load_base(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,                         # rank: higher = more capacity, more memory
        lora_alpha=alpha,            # scale of the adapter's influence
        lora_dropout=0.05,
        target_modules=default_targets(model),
    )
    model = get_peft_model(model, lora_config)      # base frozen, adapters trainable
    model.print_trainable_parameters()

    features = build_features(load_pairs(data_file), tokenizer)
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=adapter_dir + "_runs",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,                           # optimizer: default AdamW, applied to the adapters only
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        use_cpu=not use_cuda,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ListDataset(features),
                      data_collator=PadCollator(tokenizer.pad_token_id))
    print("Training...")
    trainer.train()

    model.save_pretrained(adapter_dir)              # saves only the adapter (small)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(adapter_dir, "base_model.txt"), "w") as f:
        f.write(model_name)
    print(f"Adapter saved to {adapter_dir}/")


# ============================================================
# 5. RESPONSE: reload the saved adapter and answer
# ============================================================

def load_tuned(adapter_dir):
    """What you do later, in another program: base model + saved adapter."""
    base_name = open(os.path.join(adapter_dir, "base_model.txt")).read().strip()
    base, tokenizer = load_base(base_name)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def answer(model, tokenizer, question, use_adapter=True, max_new_tokens=60):
    inputs = tokenizer(PROMPT_TEMPLATE.format(q=question), return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**inputs, **kwargs)
        else:
            with model.disable_adapter():           # same model with the adapter switched off = the original
                out = model.generate(**inputs, **kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).split("###")[0].strip()


def compare(model, tokenizer, questions):
    for q in questions:
        print(f"\nQ: {q}")
        print(f"  original   : {answer(model, tokenizer, q, use_adapter=False)}")
        print(f"  fine-tuned : {answer(model, tokenizer, q, use_adapter=True)}")


def chat(model, tokenizer):
    print("\nAsk your own questions (empty line to quit).")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            break
        print("Model:", answer(model, tokenizer, q))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--data", default=DATA_FILE)
    p.add_argument("--adapter", default=ADAPTER_DIR)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--ask", action="store_true", help="skip training; load the saved adapter and chat")
    a = p.parse_args()

    if not a.ask:
        train(a.model, a.epochs, a.batch, a.lr, a.r, a.alpha, a.data, a.adapter)

    print("\nReloading the saved adapter from disk...")
    model, tokenizer = load_tuned(a.adapter)
    print("\n--- Original vs fine-tuned ---")
    compare(model, tokenizer, TEST_QUESTIONS)
    if a.ask:
        chat(model, tokenizer)
