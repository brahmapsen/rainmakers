# Muse 2 BCI Agent

FastAPI microservice that listens to **Mind Monitor** OSC (Muse 2/2S) and exposes
signals + derived metrics to the Internet-of-Agents demo.

## Endpoints

- `GET /meta` – agent manifest & tool list
- `POST /invoke` – call tools:
  - `start` `{ bind_ip?: "0.0.0.0" }`
  - `stop`
  - `snapshot` `{ seconds?: 180 }`
  - `calibrate_start`
  - `calibrate_stop`
  - `demo_trigger` `{ action: "recap_last_minute"|"breathing_coach"|"focus_mode_on" }`
- `GET /snapshot?seconds=180`
- `GET /events` – SSE stream (optional)

## Run locally

uv pip install -r agents/muse/requirements.txt
uv run uvicorn agents.muse.main:app --port 8001 --reload

## smoke test
===========
curl http://127.0.0.1:8001/meta

curl -X POST http://127.0.0.1:8001/invoke -H 'content-type: application/json' \
  -d '{"name":"start"}'
curl http://127.0.0.1:8001/snapshot?seconds=30
