"""
reward_model_example.py -- TRL's RewardTrainer: train a model that SCORES answers
=================================================================================
SFT teaches what to say. DPO teaches which of two answers is better.
RewardTrainer builds a separate "judge" model: give it an answer, it returns a number
(higher = better). It learns from the same chosen/rejected pairs DPO uses.

Where the judge is used:
  - classic RLHF: the score is the reward that trains the main model (PPO)
  - ranking: generate several answers, keep the highest-scoring one
  - automatic quality checks, like a learned version of groundedness_check()

The model here is a classifier with ONE output (the score), not a text generator.

REQUIREMENTS: NVIDIA GPU (free Colab T4 is fine).
    pip install trl peft datasets transformers accelerate torch
"""

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# Plain-text pairs: the question plus a good answer vs the question plus a bad one.
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


def build_dataset() -> Dataset:
    return Dataset.from_list([
        {"chosen": f"Question: {q}\nAnswer: {good}", "rejected": f"Question: {q}\nAnswer: {bad}"}
        for q, good, bad in PAIRS
    ])


def train(model, tokenizer, dataset, output_dir="policy_reward_model", epochs=3):
    args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=256,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        # SEQ_CLS (not CAUSAL_LM): this model outputs a score, not text.
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS"),
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer.model


def score(model, tokenizer, text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**inputs).logits[0, 0].item()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)  # 1 output = the score
    model.config.pad_token_id = tokenizer.pad_token_id

    model = train(model, tokenizer, build_dataset())

    good = "Question: How many days of annual leave?\nAnswer: All full-time employees receive 20 days."
    bad = "Question: How many days of annual leave?\nAnswer: Unlimited days, whenever you like."
    print("good answer score:", score(model, tokenizer, good))
    print("bad answer score: ", score(model, tokenizer, bad))
