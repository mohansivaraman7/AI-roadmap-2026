# Own tiny LLM: build, run, host

## 1. Build and run locally (any laptop, no GPU)
    pip install -r requirements.txt
    python make_corpus.py        # or put your own text in corpus.txt
    python train.py              # about 3-4 minutes on CPU; writes checkpoint/ (~3.5 MB)
    streamlit run app.py         # opens http://localhost:8501

## 2. What you are hosting
Only two small files: `checkpoint/model.pt` (weights + config) and `checkpoint/tokenizer.json`,
plus `model.py` (the architecture), `app.py` (the UI) and `requirements.txt`.
A model this size runs on a free CPU host. No GPU is needed to SERVE it.

## 3. Where to host

| Option | Cost | Best for | How |
|---|---|---|---|
| Streamlit Community Cloud | Free | Quickest public demo | Push this folder to a GitHub repo, then at share.streamlit.io choose the repo and `app.py` |
| Hugging Face Spaces | Free CPU tier | AI audience, easy sharing | New Space, SDK = Docker, upload these files (the Dockerfile listens on port 7860) |
| Render / Railway / Fly.io | Free to low paid | A normal web service with your own domain | Connect the GitHub repo, use the Dockerfile |
| Google Cloud Run / AWS / Azure | Pay per use | Company, internal use | `docker build` and deploy the container |
| Your own server or office PC | Free | Private use on your network | `streamlit run app.py --server.address 0.0.0.0` and share the IP:8501 |

Free tiers sleep when idle and can change their limits, so check the host's current terms.

## 4. Larger models
- Weights above roughly 50-100 MB do not belong in a normal git repo. Use Git LFS, or upload to the
  Hugging Face Hub and download at startup with `hf_hub_download`.
- Above about 1B parameters you need a GPU host for acceptable speed (Hugging Face GPU Spaces,
  RunPod, Modal, or a cloud GPU VM), usually served through vLLM or Ollama instead of a plain script.

## 5. Before sharing publicly
- This model only continues text in the style of its training data. It does not follow instructions and can state wrong facts confidently.
- Do not train on private or confidential documents if the model or the app will be public.
