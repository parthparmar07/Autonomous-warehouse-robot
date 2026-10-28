FROM python:3.13-slim
WORKDIR /warehouse
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY frontend ./frontend
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
