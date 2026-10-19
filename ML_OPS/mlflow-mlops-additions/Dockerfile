FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
# whenever only app code changes, not requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy what the running API needs, plus train_model.py so this same
# image can also be used as docker-compose's one-off "train" service.
COPY app.py train_model.py house_model.pkl ./

EXPOSE 5000

# gunicorn (already in requirements.txt) instead of Flask's dev server —
# this mirrors how the app is meant to run in production on Render.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
