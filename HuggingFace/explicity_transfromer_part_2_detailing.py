import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

model_name = "distilgpt2"

# --------------------------------
# 1. Load tokenizer
# --------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

# --------------------------------
# 2. Load pretrained Transformer
# --------------------------------

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

# --------------------------------
# 3. Input
# --------------------------------

text = "Python is"

# --------------------------------
# 4. Tokenize
# --------------------------------

inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("Input IDs:")
print(inputs["input_ids"])

# --------------------------------
# 5. Run Transformer
# --------------------------------

outputs = model(**inputs)

# --------------------------------
# 6. Logits
# --------------------------------

logits = outputs.logits

print("\nLogits shape:")
print(logits.shape)

# --------------------------------
# 7. Generate
# --------------------------------

generated = model.generate(
    **inputs,
    max_new_tokens=30
)

# --------------------------------
# 8. Decode
# --------------------------------

result = tokenizer.decode(
    generated[0],
    skip_special_tokens=True
)

print("\nGenerated:")
print(result)