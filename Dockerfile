# ModelSearch Demo — Hugging Face Spaces (Docker)
# Single-server: backend serves UI + API on port 7860 when SERVE_UI=1.

FROM python:3.10-slim

# HF Spaces run as user 1000; create user first, then install/copy as that user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

RUN pip install --no-cache-dir --upgrade pip

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY --chown=user . .

# Create dirs backend may write to (jobs, etc.)
RUN mkdir -p data/jobs

ENV SERVE_UI=1 PORT=7860
EXPOSE 7860

# Gunicorn: 1 worker to avoid subprocess/thread issues; long timeout for search pipeline
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "600", "src.demo.backend:app"]
