import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. TRAINING DATA
# ============================================================

sentences = [
    "i love cricket",
    "i love football",
    "cricket is great",
    "football is great",
    "i like cricket",
    "i like football",

    "i hate cricket",
    "i hate football",
    "cricket is boring",
    "football is boring",
    "i dislike cricket",
    "i dislike football"
]

# 1 = Positive
# 0 = Negative

labels = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
]


# ============================================================
# 2. CREATE A SIMPLE VOCABULARY
# ============================================================

vocab = {
    "<PAD>": 0,
    "i": 1,
    "love": 2,
    "hate": 3,
    "like": 4,
    "dislike": 5,
    "cricket": 6,
    "football": 7,
    "is": 8,
    "great": 9,
    "boring": 10
}

print("Vocabulary:")
print(vocab)


# ============================================================
# 3. TOKENIZER
# ============================================================

def tokenize(sentence):

    words = sentence.lower().split()

    token_ids = []

    for word in words:
        token_ids.append(vocab[word])

    return token_ids


# Convert sentences into token IDs

tokenized_sentences = []

for sentence in sentences:

    tokens = tokenize(sentence)

    tokenized_sentences.append(tokens)


print("\nTokenized sentences:")

for sentence, tokens in zip(sentences, tokenized_sentences):

    print(sentence)
    print(tokens)


# ============================================================
# 4. CONVERT TO PYTORCH TENSOR
# ============================================================

X = torch.tensor(tokenized_sentences)

y = torch.tensor(labels)

print("\nInput tensor:")
print(X)

print("\nLabels:")
print(y)


# ============================================================
# 5. CREATE TRANSFORMER MODEL
# ============================================================

class TinyTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=16
        )

        # ----------------------------------------------------
        # Self-Attention
        # ----------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=16,
            num_heads=2,
            batch_first=True
        )

        # ----------------------------------------------------
        # Layer Normalization
        # ----------------------------------------------------

        self.norm1 = nn.LayerNorm(16)

        # ----------------------------------------------------
        # Feed Forward Network
        # ----------------------------------------------------

        self.ffn = nn.Sequential(

            nn.Linear(16, 32),

            nn.ReLU(),

            nn.Linear(32, 16)
        )

        # ----------------------------------------------------
        # Second Layer Normalization
        # ----------------------------------------------------

        self.norm2 = nn.LayerNorm(16)

        # ----------------------------------------------------
        # Final Classification Layer
        # ----------------------------------------------------

        self.classifier = nn.Linear(16, 2)


    def forward(self, x, return_attention=False):

        # ====================================================
        # STEP 1: TOKEN EMBEDDING
        # ====================================================

        x = self.embedding(x)

        # Shape:
        #
        # [batch_size, sequence_length, embedding_size]
        #
        # Example:
        #
        # [12, 3, 16]


        # ====================================================
        # STEP 2: SELF-ATTENTION
        # ====================================================

        attention_output, attention_weights = self.attention(
            x,      # Query
            x,      # Key
            x       # Value
        )

        # ====================================================
        # STEP 3: ADD & NORM
        # ====================================================

        x = self.norm1(
            x + attention_output
        )


        # ====================================================
        # STEP 4: FEED FORWARD NETWORK
        # ====================================================

        ffn_output = self.ffn(x)


        # ====================================================
        # STEP 5: ADD & NORM
        # ====================================================

        x = self.norm2(
            x + ffn_output
        )


        # ====================================================
        # STEP 6: CREATE SENTENCE REPRESENTATION
        # ====================================================

        # Take the average of all token representations

        x = x.mean(dim=1)


        # ====================================================
        # STEP 7: CLASSIFICATION
        # ====================================================

        output = self.classifier(x)


        if return_attention:

            return output, attention_weights

        return output


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = TinyTransformer()

print("\n\nMODEL:")
print(model)


# ============================================================
# 7. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# ============================================================
# 9. TRAINING
# ============================================================

epochs = 500

print("\n\nTRAINING...\n")

for epoch in range(epochs):

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(X)

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = criterion(output, y)

    # --------------------------------------------------------
    # Clear old gradients
    # --------------------------------------------------------

    optimizer.zero_grad()

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Update weights
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    if (epoch + 1) % 50 == 0:

        predictions = torch.argmax(
            output,
            dim=1
        )

        accuracy = (
            predictions == y
        ).float().mean()

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item() * 100:.2f}%"
        )


# ============================================================
# 10. TEST THE MODEL
# ============================================================

def predict(sentence):

    model.eval()

    tokens = tokenize(sentence)

    input_tensor = torch.tensor(
        [tokens]
    )

    with torch.no_grad():

        output, attention = model(
            input_tensor,
            return_attention=True
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()


    if prediction == 1:

        result = "POSITIVE"

    else:

        result = "NEGATIVE"


    print("\n--------------------------------")

    print("Sentence:")
    print(sentence)

    print("Prediction:")
    print(result)

    print("Probabilities:")

    print(
        "Negative:",
        round(
            probabilities[0][0].item(),
            4
        )
    )

    print(
        "Positive:",
        round(
            probabilities[0][1].item(),
            4
        )
    )

    print("--------------------------------")

    return attention


# ============================================================
# 11. TEST SENTENCES
# ============================================================

attention1 = predict(
    "i love cricket"
)

attention2 = predict(
    "i hate football"
)

attention3 = predict(
    "cricket is great"
)

attention4 = predict(
    "football is boring"
)