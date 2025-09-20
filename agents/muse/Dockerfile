# agents/muse_bci/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
ENV OSC_PORT=5000 WINDOW_SEC=180 CAL_SEC=15 EMA_ALPHA=0.30 MOVE_GATE=0.03
EXPOSE 8001/tcp 5000/udp
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
