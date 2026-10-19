"""
qlora_finetune.py -- SAME goal as lora_finetune.py, but the base model
is loaded in 4-bit (quantized) first, to demonstrate the memory-saving
trick that makes QLoRA able to fine-tune much larger models than plain
LoRA on the same hardware.

REQUIRES AN NVIDIA GPU. bitsandbytes (the 4-bit quantization library)
only supports CUDA -- this will NOT run on a Mac (no CUDA) or on CPU.
If you're on a Mac, this script is here to READ and understand the
difference from lora_finetune.py, not to run locally -- use Google
Colab with a GPU runtime instead.

SETUP:
    pip install torch transformers peft datasets accelerate bitsandbytes --break-system-packages

RUN (after create_training_data.py has produced policy_training_data.jsonl):
    python qlora_finetune.py
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

BASE_MODEL = "distilgpt2"
OUTPUT_DIR = "policy_qlora_adapter"


# ============================================================
# 1. QUANTIZATION CONFIG -- THIS is the "Q" in QLoRA
# ============================================================
# Loads the base model's weights in 4-bit instead of the usual 16/32-bit,
# shrinking its memory footprint dramatically before any LoRA adapter is
# even added. This is the ONLY conceptual difference from lora_finetune.py
# -- everything after this section follows the same pattern.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # "normal float 4" -- a quantization format tuned for LLM weights
    bnb_4bit_compute_dtype=torch.float16,  # compute still happens in float16 for accuracy
    bnb_4bit_use_double_quant=True,       # quantizes the quantization constants too, for extra savings
)


# ============================================================
# 2. LOAD THE BASE MODEL IN 4-BIT
# ============================================================

print("Loading base model in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",   # automatically places the model on the available GPU
)

model = prepare_model_for_kbit_training(model)  # required prep step before adding LoRA to a quantized model


# ============================================================
# 3. WRAP WITH LoRA -- IDENTICAL to lora_finetune.py from here
# ============================================================

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============================================================
# 4. LOAD AND TOKENIZE THE TRAINING DATA
# ============================================================

dataset = load_dataset("json", data_files="policy_training_data.jsonl", split="train")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])


# ============================================================
# 5. TRAIN
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    fp16=True,
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
# 6. SAVE THE ADAPTER
# ============================================================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"QLoRA adapter saved to {OUTPUT_DIR}/")


# ============================================================
# 7. QUICK TEST
# ============================================================

print("\nTesting the fine-tuned model:")
prompt = "### Question:\nHow many days of annual leave do employees get?\n\n### Answer:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
