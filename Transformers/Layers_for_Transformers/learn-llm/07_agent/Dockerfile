FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent.py app.py ./
EXPOSE 8000
# API key is passed at runtime:  docker run -e ANTHROPIC_API_KEY=sk-... -p 8000:8000 tutor-agent
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
