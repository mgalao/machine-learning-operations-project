FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-serving.txt

COPY app/ /app/app/
COPY data/06_models/ /app/models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# FROM python:3.11-slim

# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     python3-dev \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*


# WORKDIR /app

# RUN pip install uv
# COPY requirements.txt .
# RUN uv pip install --no-cache-dir -r requirements.txt --system

# COPY . .

# EXPOSE 8000
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

