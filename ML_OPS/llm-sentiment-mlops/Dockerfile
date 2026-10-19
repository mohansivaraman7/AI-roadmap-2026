# Small, CPU-only image for serving the fine-tuned sentiment model.
FROM python:3.11-slim

WORKDIR /app

# Install deps first so Docker can cache this layer across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY mlruns/ mlruns/
COPY mlartifacts/ mlartifacts/

ENV CONFIG_PATH=configs/config.yaml
EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
