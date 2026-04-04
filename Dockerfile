FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    torch_geometric \
    sentence-transformers \
    faiss-cpu \
    "ray[serve]" \
    pyyaml \
    pandas \
    numpy \
    fastparquet \
    python-dotenv \
    tqdm
COPY configs/ configs/
COPY service/ service/

COPY data/embeddings/chunk_embeddings.npy data/embeddings/chunk_embeddings.npy
COPY data/embeddings/chunk_ids.npy        data/embeddings/chunk_ids.npy
COPY data/embeddings/faiss.index         data/embeddings/faiss.index
COPY data/graph/edges.parquet            data/graph/edges.parquet
COPY data/processed/corpus.parquet      data/processed/corpus.parquet
COPY checkpoints/ checkpoints/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "service/serve.py"]
