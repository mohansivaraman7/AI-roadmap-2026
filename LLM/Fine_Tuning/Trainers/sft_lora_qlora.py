"""
sft_lora_qlora.py -- LoRA or QLoRA fine-tuning with TRL's SFTTrainer (full program)
====================================================================================
Same goal as lora_finetune.py / qlora_finetune.py: teach a small model the company
policy facts as question -> answer pairs. The difference is HOW the training is run.

  Earlier scripts (plain Trainer)            This script (TRL SFTTrainer)
  ---------------------------------          ----------------------------------
  write a tokenize() function                SFTTrainer tokenizes the "text" column
  build a DataCollatorForLanguageModeling    handled for you
  call get_peft_model() yourself             pass peft_config=lora_config
  Trainer(...)                               SFTTrainer(...)

ONE SCRIPT, TWO MODES:
    python sft_lora_qlora.py --mode lora      # base model in normal precision
    python sft_lora_qlora.py --mode qlora     # base model loaded in 4-bit (NVIDIA GPU only)

REQUIREMENTS: an NVIDIA GPU. Use a free Google Colab T4 if you are on a Mac.
    pip install trl peft datasets transformers accelerate torch
    pip install bitsandbytes                  # only for --mode qlora

Output: a small adapter folder (a few MB), not a full model copy.
"""

import argparse
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-0.5B"   # small, and uses q_proj/v_proj names like Llama, Mistral, Gemma

# Same facts as company_policy.pdf, shaped as question -> answer pairs
EXAMPLES = [
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

PROMPT_TEMPLATE = "### Question:\n{q}\n\n### Answer:\n"


def build_dataset(tokenizer) -> Dataset:
    # SFTTrainer reads the "text" column. EOS at the end teaches the model WHEN TO STOP.
    rows = [{"text": PROMPT_TEMPLATE.format(q=q) + a + tokenizer.eos_token} for q, a in EXAMPLES]
    return Dataset.from_list(rows)


def load_model_and_tokenizer(mode: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "qlora":
        # The "Q": load the frozen base model in 4-bit to save GPU memory.
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
        # SFTTrainer does NOT do this step for you, so call it yourself in QLoRA mode.
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    return model, tokenizer


def make_lora_config(mode: str) -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # QLoRA setups usually train every linear layer; LoRA often just q and v.
        target_modules="all-linear" if mode == "qlora" else ["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )


def train(model, tokenizer, dataset, mode: str, output_dir: str, epochs: int = 10):
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",     # which column holds the training text
        max_length=256,                # truncate long examples
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=make_lora_config(mode),   # SFTTrainer wraps the model with LoRA for you
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer.model


def ask(model, tokenizer, question: str, max_new_tokens: int = 40) -> str:
    inputs = tokenizer(PROMPT_TEMPLATE.format(q=question), return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--epochs", type=int, default=10)
    cli = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no NVIDIA GPU found. This will be very slow, and --mode qlora will fail.")

    model, tokenizer = load_model_and_tokenizer(cli.mode)
    dataset = build_dataset(tokenizer)
    output_dir = f"policy_sft_{cli.mode}_adapter"

    model = train(model, tokenizer, dataset, cli.mode, output_dir, cli.epochs)

    print("\n--- Test ---")
    print(ask(model, tokenizer, "How many days of annual leave do employees get?"))
    print(f"\nAdapter saved to {output_dir}/")
