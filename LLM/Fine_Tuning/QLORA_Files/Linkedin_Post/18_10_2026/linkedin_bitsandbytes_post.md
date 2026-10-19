# How a 7B model fits on a small GPU: BitsAndBytesConfig explained

A 7-billion-parameter model needs about 14 GB just to hold its weights in normal 16-bit form. Many GPUs cannot do that, and fine-tuning needs even more.

QLoRA's answer is to store the frozen base model in 4 bits, about 3.5 GB for the same weights, and train only small LoRA adapters on top. In Hugging Face, that "Q" is four lines:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

**What each line does**
- `load_in_4bit`: store the weights in 4 bits instead of 16.
- `nf4`: a 4-bit format designed for LLM weights. They are not evenly spread, so its 16 levels sit where the weights actually are.
- `double_quant`: also compresses the scaling constants, for a small extra saving.
- `compute_dtype`: weights are stored in 4-bit but expanded to 16-bit for each calculation, so quality holds up.

Then you add LoRA adapters, and only those get trained.

**Two things I learned teaching this**
1. It needs an NVIDIA GPU. bitsandbytes does not run on a Mac or CPU. A free Colab T4 is enough to try it.
2. The 14 GB and 3.5 GB figures are weights only. Training also needs memory for activations and the optimizer, so plan for more.

If you teach or learn AI, build it once yourself: manual loop, then LoRA, then QLoRA. You will understand what each library is hiding.

#AI #LLM #FineTuning #QLoRA #MachineLearning
