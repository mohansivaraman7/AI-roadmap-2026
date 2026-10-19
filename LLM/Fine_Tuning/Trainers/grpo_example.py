"""
grpo_example.py -- TRL's GRPOTrainer: learn from a reward FUNCTION you write
============================================================================
No example answers (SFT) and no answer pairs (DPO). Instead:
  1. the model generates several answers to each question
  2. YOUR Python function scores each answer
  3. answers that scored above the group average get reinforced

Best when an answer can be checked by code: maths, code that must pass tests,
exact numbers from a document, required output format. This is how many
"reasoning" models are trained.

Here the reward checks that the answer contains the correct policy number and is short.
This is a teaching-size demo: a 0.5B model with 4 questions will not become reliable.

REQUIREMENTS: NVIDIA GPU (free Colab T4 is fine).
    pip install trl peft datasets transformers accelerate torch
"""

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

QUESTIONS = [
    ("How many days of annual leave do employees get?", "20"),
    ("How many sick leave days do employees get per year?", "10"),
    ("How many days per week can employees work from home?", "2"),
    ("Up to how many days of annual leave can be carried forward?", "5"),
]


def build_dataset() -> Dataset:
    # The extra "answer" column is passed to the reward function automatically.
    return Dataset.from_list([
        {"prompt": f"### Question:\n{q}\n\n### Answer:\n", "answer": a} for q, a in QUESTIONS
    ])


def reward_correct_and_short(prompts, completions, answer, **kwargs):
    """Return one score per completion. TRL calls this for you."""
    rewards = []
    for completion, correct in zip(completions, answer):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        score = 1.0 if correct in text.split() or f"{correct} " in text else 0.0   # right number mentioned
        score += 0.3 if len(text.split()) <= 25 else 0.0                           # and keeps it short
        rewards.append(score)
    return rewards


def train(model, tokenizer, dataset, output_dir="policy_grpo_adapter", epochs=1):
    args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        num_generations=4,            # answers sampled per question; batch size must divide evenly by this
        max_completion_length=48,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_correct_and_short,    # your function, not a model
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
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
    train(model, tokenizer, build_dataset())
    print("GRPO adapter saved to policy_grpo_adapter/")
