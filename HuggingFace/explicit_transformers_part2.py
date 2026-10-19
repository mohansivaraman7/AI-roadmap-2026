from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

text = "Python is amazing"

tokens = tokenizer.tokenize(text)

print(tokens)

ids = tokenizer.encode(text)

print(ids)


inputs = tokenizer(
    text,
    return_tensors="pt"
)

print(inputs)

outputs = model(**inputs)

import torch

logits = outputs.logits

last_logits = logits[:, -1, :]

probabilities = torch.softmax(
    last_logits,
    dim=-1
)

next_token = torch.argmax(
    probabilities,
    dim=-1
)

print(next_token)


next_word = tokenizer.decode(next_token)

print(next_word)

output = model.generate(
    **inputs,
    max_new_tokens=30
)

print(
    tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
)

