# AI Roadmap 2026

Course material, notes and hands-on code for a complete AI / ML learning path — from SQL and classical machine learning through deep learning, transformers, LLMs, RAG, AI agents and MLOps.

Prepared by **Mohan Sivaraman** — Machine Learning Engineer, AI Engineer and Python SME.

---

## Learning path

The material follows the type of data each family of models works with:

| Stage | Data | Focus | Folder |
|---|---|---|---|
| 1 | Structured (CSV, Excel, SQL) | Machine Learning | `DB`, `Machine_Learning` |
| 2 | Unstructured (images, video, audio) | Deep Learning | `Neural Networks`, `Deep_Learning` |
| 3 | Semi-structured (text) | NLP → Transformers → LLMs | `Transformers`, `HuggingFace`, `LLM` |
| 4 | Your own knowledge | RAG, embeddings, vector DBs | `RAG_Types`, `Embedding_Vector_DB` |
| 5 | Tools and autonomy | AI agents, MCP, multi-agent | `AI_Agent`, `LLM/Multi_Agent` |
| 6 | Production | APIs, Docker, Kubernetes, MLOps | `API_and_Deployment`, `Docker_and_Kubernetes`, `ML_OPS` |

**Stack:** Python · scikit-learn · PyTorch · TensorFlow · Hugging Face · LangChain / LangGraph / LangSmith · FAISS / ChromaDB · Streamlit · Flask / FastAPI · Docker · Kubernetes · MLflow / Kubeflow

---

## Repository structure

### Foundations
- **`DB/`** — SQL guide, joins, HAVING, stored procedures, and loading MySQL data into pandas (with a sample chocolate-sales database).
- **`DataSet/`** — Datasets used across the course (IPL tweets, cancer, Swiggy, World Cup, social media users, network data, and more).

### Machine Learning
- **`Machine_Learning/`** — Preprocessing, outliers, linearity and correlation, logistic regression, regularization, KNN, SVM, K-Means, hierarchical clustering, PCA, curse of dimensionality, bagging, boosting and stacking, column-transformer pipelines, time series (ARIMA) and model-selection tips.

### Deep Learning
- **`Neural Networks/`** — Markdown notes: activation functions, backpropagation, gradient descent, CNNs, autoencoders, GANs, evaluation metrics, hyperparameter tuning.
- **`Deep_Learning/`** — ANNs and CNNs with PyTorch, data loaders, epochs, image processing, RNNs, video processing and classification, plus end-to-end projects (motion detection, CNN-LSTM, video classification).

### Transformers and LLMs
- **`Transformers/`** — Self-attention, transformer layers, a transformer built from scratch and a tiny GPT with an agent (notebooks).
- **`HuggingFace/`** — Tokenisation, Auto models and working with pretrained transformer models.
- **`LLM/`** — Building an LLM, fine-tuning, prompt engineering, harness engineering, LangChain, RAG, RAG + MCP integration and multi-agent systems.
- **`GENAI/`** — Generative AI ecosystem overview and an image generator.

### RAG and Agents
- **`RAG_Types/`** — Eight runnable RAG patterns:
  1. Simple RAG
  2. Multi-query RAG
  3. Hybrid search RAG
  4. Re-ranking RAG
  5. HyDE RAG
  6. Corrective RAG
  7. Graph RAG
  8. Agentic RAG
- **`Embedding_Vector_DB/`** — Creating embeddings and answering questions from a vector store.
- **`AI_Agent/`** — AI foundations and types of agent systems (visual guides).

### Deployment and MLOps
- **`API_and_Deployment/`** — Serving a house-price model as an API and deploying with Flask on Render (training manual included).
- **`ai_ml/`** — A small train → pickle → serve → test example app.
- **`Docker_and_Kubernetes/`** — Docker, Kubernetes, Kubeflow and Jenkins CI/CD guide.
- **`ML_OPS/`** — MLflow, data versioning, an LLM sentiment-analysis MLOps pipeline and a complete ML API project.

### Course delivery
- **`Crash_Course/`** — Day-by-day crash-course sessions (August–September 2026), including MCP with databases and harness engineering.
- **`Projects/`** — Capstone projects: network intrusion detection, predictive maintenance API and an ML/DL project portfolio.
- **`Assessment/`** — Student assessments.
- **`Syllabus/`** — Course schedule.
- **`Mohan's_New_AI_Batch/`**, **`Mohan_AI_Batch`** — Roadmap notes for the new AI batch.

---

## Getting started

```bash
git clone https://github.com/mohansivaraman7/AI-roadmap-2026.git
cd AI-roadmap-2026

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Some folders (for example `RAG_Types`, `HuggingFace`, `ai_ml`) have their own `requirements.txt` — install those when working in that folder. Examples that call hosted LLMs need your own API keys set as environment variables; never commit keys to the repo.

---

## Notes

- `DataSet/flights_delay.csv` is not included because it exceeds GitHub's 100 MB file limit.
- Virtual environments, IDE settings and cache files are excluded via `.gitignore`.
