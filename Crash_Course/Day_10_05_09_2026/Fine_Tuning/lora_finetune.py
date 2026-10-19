"""
lora_finetune.py -- fine-tunes a small model on policy_training_data.jsonl
using LoRA, so the model's own weights learn these facts, instead of
retrieving them at query time like your RAG pipeline did.

REQUIRES A GPU. Unlike everything else in this course (Ollama, FAISS,
Streamlit, MCP), fine-tuning needs real GPU compute -- on CPU this would
take hours even for this tiny example. Use a machine with an NVIDIA GPU,
or run this in Google Colab with a free GPU runtime (Runtime -> Change
runtime type -> GPU).

SETUP:
    pip install torch transformers peft datasets accelerate --break-system-packages

RUN (after create_training_data.py has produced policy_training_data.jsonl):
    python lora_finetune.py
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "distilgpt2"          # small, so this runs quickly even on a modest GPU
OUTPUT_DIR = "policy_lora_adapter"  # where the trained adapter gets saved


# ============================================================
# 1. LOAD THE BASE MODEL AND TOKENIZER
# ============================================================
# Notice: this is the SAME distilgpt2 you saw fail earlier in this course
# (it couldn't follow "answer only from context" instructions). LoRA
# fine-tuning is one real way to actually improve that -- instead of just
# hoping a bigger prompt fixes it, you teach the model directly.

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token  # distilgpt2 has no pad token by default

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)


# ============================================================
# 2. WRAP THE MODEL WITH A LoRA ADAPTER
# ============================================================
# The base model's weights are left untouched (frozen). Only these small
# adapter matrices get trained -- a tiny fraction of the model's total
# parameters.

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                 # rank of the adapter matrices -- higher = more capacity, more memory
    lora_alpha=16,        # scaling factor for the adapter's influence
    lora_dropout=0.05,
    target_modules=["c_attn"],  # which layers get adapters (attention layers, for GPT-2 architecture)
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # prints how few parameters are actually being trained


# ============================================================
# 3. LOAD AND TOKENIZE THE TRAINING DATA
# ============================================================

dataset = load_dataset("json", data_files="policy_training_data.jsonl", split="train")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])


# ============================================================
# 4. TRAIN
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,          # small dataset, so several passes over it
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",           # we'll save the adapter manually below
    report_to=[],                 # disable wandb/etc. logging
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Training...")
trainer.train()


# ============================================================
# 5. SAVE THE ADAPTER (small -- a few MB, not a full model copy)
# ============================================================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}/")


# ============================================================
# 6. QUICK TEST -- ask it something from the training data
# ============================================================

print("\nTesting the fine-tuned model:")
prompt = "### Question:\nHow many days of annual leave do employees get?\n\n### Answer:\n"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
