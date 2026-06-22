FROM python:3.12-slim

# Single-threaded BLAS for determinism + speed (see ensemblekit/__init__.py).
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run the full offline benchmark and print the dissociation table.
CMD ["python", "-m", "evals.harness"]
