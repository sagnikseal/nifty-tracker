# NIFTY v4 engine: runs anywhere Docker runs (Windows/macOS/Linux/cloud VM).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Kolkata NE_DATA_DIR=/data
WORKDIR /app
COPY requirements-engine.txt .
RUN pip install --no-cache-dir -r requirements-engine.txt
COPY nifty_engine ./nifty_engine
COPY config ./config
COPY tests ./tests
VOLUME ["/data"]
EXPOSE 8010
HEALTHCHECK --interval=5m --timeout=10s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8010/api/v4/health')"
# First boot: download history + train if no model yet, then run the scheduler + dashboard.
CMD ["sh", "-c", "[ -f /data/models/champion.json ] || python -m nifty_engine all; python -m nifty_engine run"]
