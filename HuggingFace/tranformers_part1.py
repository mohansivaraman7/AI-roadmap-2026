from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

result = generator(
    "Python is",
    max_new_tokens=10
)

print(result)