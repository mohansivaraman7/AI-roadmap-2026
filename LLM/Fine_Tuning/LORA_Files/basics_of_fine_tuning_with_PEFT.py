# pip install peft
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
config = LoraConfig(
    r=8,                                  # rank of the adapter matrices
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # which layers get adapters
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()   # e.g. trainable: ~0.1%