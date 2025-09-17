## README.md (quickstart)

# Date read from Muse 2 (EEG, PLG, ACC, GYRO) - OSC: Open Sound Control, LSL: Lab Streaming Layer

## 0) Install deps with uv
uv sync

## 1) Start the Agent
uv run uvicorn src.agent_orchestrator:app --port 8001 --reload

## 2) Start the Synthetic EEG streamer (LSL + control API)
uv run python src/synth_lsl.py
# It serves control API at http://127.0.0.1:7000

## 3) Start the Receiver (features + classification → POST to agent)
uv run python src/receiver.py [if you are using synthetic data]
uv run python src/osc_receiver.py


## 4) Start UI
uv run streamlit run ui.py


# New terminal
curl -X POST http://127.0.0.1:7000/state/relaxed
curl -X POST http://127.0.0.1:7000/state/engaged

You should see the receiver print alpha/beta ratio changes and the agent logging different actions.

## Notes
- This sim reproduces realistic alpha/beta dynamics so your downstream Agent/LLM glue stays identical when you swap to real hardware (Muse/OpenBCI/EMOTIV) via LSL/BrainFlow.
- To integrate a real LLM: in `agent_orchestrator.py`, send `evt.model_dump()` into your model/tooling and branch actions based on its plan.
```

### Installing pylsl for Ubuntu
lsb_release -cs

VER=1.16.2
CODENAME=<codename>   # noble or jammy
curl -fLO "https://github.com/sccn/liblsl/releases/download/v${VER}/liblsl-${VER}-${CODENAME}_amd64.deb"

# Install it, then refresh linker cache
sudo dpkg -i "liblsl-${VER}-${CODENAME}_amd64.deb" || sudo apt-get -f install -y
sudo ldconfig

uv run python - <<'PY'
from pylsl import lib
print("LSL found:", lib)
PY