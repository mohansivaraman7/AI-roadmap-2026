"""
dpo_lora_example.py -- TRL's DPOTrainer: teach a model WHICH answer is better
=============================================================================
SFTTrainer teaches a model WHAT to say, from example answers.
DPOTrainer (Direct Preference Optimization) teaches it WHICH of two answers is
better, from pairs of a "chosen" (good) answer and a "rejected" (bad) answer.

Typical uses: make the model refuse to guess, keep answers short, stay polite,
admit "I don't know", follow a house style.

Data format: three columns -- prompt, chosen, rejected.

USUAL ORDER IN PRACTICE:  1) SFT first (learn the task)  ->  2) DPO second (polish behaviour)

REQUIREMENTS: NVIDIA GPU (free Colab T4 is fine).
    pip install trl peft datasets transformers accelerate torch
"""

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# For the SAME question, one answer sticks to the policy and one invents details.
PAIRS = [
    ("How many days of annual leave do employees get?",
     "All full-time employees receive 20 days of annual leave per calendar year.",
     "Employees get 30 days, plus unlimited extra days if the manager agrees."),
    ("How many days per week can employees work from home?",
     "Employees can work from home 2 days per week, with manager approval.",
     "You can work from home every day, no approval needed."),
    ("Is there a policy on stock options?",
     "I don't have that information in the policy. Please contact HR.",
     "Yes, every employee gets 1,000 stock options after one year."),
    ("When are employees paid?",
     "Employees are paid on the last working day of each month.",
     "Salary is paid on the 15th of every month."),
]

PROMPT = "### Question:\n{q}\n\n### Answer:\n"


def build_dataset(tokenizer) -> Dataset:
    return Dataset.from_list([
        {"prompt": PROMPT.format(q=q), "chosen": good + tokenizer.eos_token, "rejected": bad + tokenizer.eos_token}
        for q, good, bad in PAIRS
    ])


def train(model, tokenizer, dataset, output_dir="policy_dpo_adapter", epochs=3):
    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=5e-5,   # DPO uses small learning rates; with LoRA, 5e-5 is a common start
        beta=0.1,                                 # how strongly to push away from the original model's habits
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=256,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                               target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        # No ref_model needed: with LoRA, the reference is the same model with the adapter switched off.
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer.model


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    train(model, tokenizer, build_dataset(tokenizer))
    print("DPO adapter saved to policy_dpo_adapter/")
