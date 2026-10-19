# From LSTM to Transformers to an Agent — a step-by-step kit

Seven small programs. Each one adds **one idea** to the previous one.
Read the comment at the top of each file, run it, then read the output.

```
LSTM ──> Attention ──> Self-Attention ──> Multi-Head ──> Transformer ──> Mini-GPT ──> Agent
 01         02             03                 04             05             06          07
```

## Setup (once)

```bash
pip install torch numpy           # steps 1–6
pip install -r 07_agent/requirements.txt   # step 7
```

## The path

| Step | File | The one idea | What you'll see when you run it |
|---|---|---|---|
| 1 | `01_lstm.py` | Read word by word, keep a memory (gates) | Hand-written LSTM cell, then an LSTM learning to remember the first number (→ 100%) |
| 2 | `02_attention.py` | Don't squeeze into one vector — look back at all words | Translation demo: each French word focuses ~91% on the right English word |
| 3 | `03_self_attention.py` | Words attend to each other: `softmax(QKᵀ/√d)·V` | A 5×5 attention matrix, then the causal (GPT) version with the future masked |
| 4 | `04_multi_head.py` | Several attentions in parallel | 4 heads, 4 different attention patterns, same output shape |
| 5 | `05_transformer_block.py` | Attention + feed-forward + residual + LayerNorm + positions | Proof that without positions "dog bites man" = "man bites dog" |
| 6 | `06_mini_gpt.py` | Predict the next token, repeat | ~100K-param GPT going from gibberish to sentences in ~30 s |
| 7 | `07_agent/` | Use a big LLM as a brain + tools + a loop, deployed as an API | Chat in the browser; agent calls calculator / glossary / notes tools |

## Key ideas to say out loud after each step

1. **LSTM** — sequential; memory fades on long text; can't parallelise.
2. **Attention** — query asks, keys answer "how relevant am I", values get mixed.
3. **Self-attention** — Q, K, V all come from the same sentence; whole sentence at once.
4. **Multi-head** — each head learns a different relationship.
5. **Transformer** — stack blocks; positions must be added because attention ignores order.
6. **GPT** — a causal Transformer trained only on "guess the next token". (Our tiny one memorises its tiny text — real LLMs do the same thing with trillions of tokens, so they generalise.)
7. **Agent** — the LLM never runs code; it *asks* for a tool, your code runs it, the result goes back. Loop until done.

## Step 7: run and deploy the agent

```bash
cd 07_agent
export ANTHROPIC_API_KEY=sk-ant-...      # get one at console.anthropic.com

python agent.py                           # 1) chat in the terminal
uvicorn app:app --port 8000               # 2) web API + chat page at http://localhost:8000

docker build -t tutor-agent .             # 3) container, deployable anywhere
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -p 8000:8000 tutor-agent
```

The Docker image runs on Render, Railway, Google Cloud Run, AWS App Runner, etc. — set
`ANTHROPIC_API_KEY` as a secret environment variable there. Change the model with `AGENT_MODEL`.

Call the API directly:

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"session_id":"me","message":"What is 37*48? Then explain self-attention."}'
```

**Add your own tool** (the best exercise): write a Python function in `agent.py`,
add it to `TOOL_FUNCS`, and describe it in `TOOLS` with `schema(...)`. That's it.

## Exercises (optional)

- Step 1: raise `SEQ` to 100. Does the LSTM still remember? (It gets much harder — that's the motivation for attention.)
- Step 3: remove `/ math.sqrt(d)` and watch the weights become all 0s and 1s.
- Step 6: replace `text` with any book chapter and increase `LAYERS` to 4.
- Step 7: before production, swap the in-memory `SESSIONS` dict for Redis or a database.
