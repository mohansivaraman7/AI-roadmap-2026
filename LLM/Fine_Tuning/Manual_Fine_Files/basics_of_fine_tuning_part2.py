from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

for layer in model.bert.encoder.layer[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

import torch
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
)

texts = [
    "I love this product", "Excellent quality and fast delivery",
    "Very happy with the service", "Amazing experience, highly recommend",
    "This is terrible", "Worst purchase ever",
    "Very poor quality, broke in a day", "I am disappointed with the service",
]
labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
model.train()

for epoch in range(15):
    outputs = model(**inputs, labels=labels)   # forward pass + loss
    loss = outputs.loss
    loss.backward()                            # compute gradients
    optimizer.step()                           # update unfrozen weights
    optimizer.zero_grad()                      # reset for next round
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# 4. Test on new sentences
model.eval()
test = ["The product is wonderful", "Really bad experience"]
with torch.no_grad():
    preds = model(**tokenizer(test, padding=True, return_tensors="pt")).logits.argmax(dim=-1)
for t, p in zip(test, preds):
    print(t, "->", "POSITIVE" if p == 1 else "NEGATIVE")

# 5. Save the fine-tuned model
model.save_pretrained("my-finetuned-bert")
tokenizer.save_pretrained("my-finetuned-bert")