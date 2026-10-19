# Unsloth version of basics_of_fine_tuning_with_QLORA.py (same model, same LoRA settings)
#
# Needs an NVIDIA GPU. Easiest: a free Google Colab T4 runtime.
# pip install unsloth        (it installs transformers, peft, trl and bitsandbytes for you)

from unsloth import FastLanguageModel   # import unsloth BEFORE transformers/peft so its patches apply

# 1. Load the base model in 4-bit. Replaces: BitsAndBytesConfig + AutoModelForCausalLM.from_pretrained
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B",   # Unsloth's mirror of Qwen/Qwen2.5-0.5B
    max_seq_length=2048,
    load_in_4bit=True,                   # QLoRA-style 4-bit loading, one flag
    dtype=None,                          # auto-picks float16/bfloat16 for your GPU
)

# 2. Add the LoRA adapters. Replaces: prepare_model_for_kbit_training + LoraConfig + get_peft_model
model = FastLanguageModel.get_peft_model(
    model,
    r=8,                                  # rank of the adapter matrices
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # or add k_proj, o_proj, gate_proj, up_proj, down_proj
    lora_dropout=0,                       # 0 is Unsloth's fastest setting (your earlier 0.05 also works)
    bias="none",
    use_gradient_checkpointing="unsloth", # Unsloth's memory-saving version
    random_state=3407,
)

model.print_trainable_parameters()

# 3. Training uses TRL's SFTTrainer (Unsloth makes it faster behind the scenes):
#
#   from trl import SFTTrainer, SFTConfig
#   trainer = SFTTrainer(
#       model=model,
#       train_dataset=dataset,                 # a dataset with a "text" column
#       args=SFTConfig(output_dir="out", per_device_train_batch_size=2, num_train_epochs=3, learning_rate=2e-4),
#   )
#   trainer.train()
