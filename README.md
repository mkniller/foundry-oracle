# Foundry Oracle

Local AI assistant for Foundry VTT.

## Setup (uv + venv)
1) Install `uv` if you do not already have it.
2) Create the venv in the repo root:
```powershell
uv venv
```
3) Sync dependencies:
```powershell
uv sync
```
4) Run the service:
```powershell
uv run python -m uvicorn service.main:app --host 127.0.0.1 --port 8000
```
