"""
api/main.py - Vanguard Operator Dashboard API

Data flow:
  PySpark Streaming  →  POST /api/v1/telemetry  →  FastAPI  →  WebSocket  →  React UI

Start with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanguard.api")

app = FastAPI(
    title="Vanguard Operator Dashboard API",
    version="1.0.0",
    description="Real-time IIoT telemetry API backed by PySpark Structured Streaming.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MachineHealth(BaseModel):
    unit_number: int = Field(..., description="Engine unit ID")
    cycle: int = Field(..., description="Current operational cycle")
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life (cycles)")
    anomaly_score: float = Field(..., ge=0.0, description="Isolation Forest anomaly score (higher = more anomalous)")
    is_anomaly: bool = Field(..., description="True when anomaly score exceeds the trained threshold")


# ---------------------------------------------------------------------------
# In-memory state (keyed by unit_number for fast lookups)
# ---------------------------------------------------------------------------

latest_telemetry: Dict[int, MachineHealth] = {}
active_connections: List[WebSocket] = []


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

async def broadcast(payload: dict) -> None:
    """Send a JSON payload to every connected WebSocket client."""
    dead: List[WebSocket] = []
    for ws in active_connections:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_connections.remove(ws)
        logger.warning("Removed stale WebSocket connection.")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
async def root():
    return {"message": "Vanguard Dashboard API is running."}


@app.get("/health", tags=["meta"])
async def health_check():
    return {"status": "healthy", "connected_clients": len(active_connections)}


@app.get("/api/v1/engines", response_model=List[MachineHealth], tags=["telemetry"])
async def get_engines():
    """Return the latest telemetry snapshot for every engine unit."""
    return list(latest_telemetry.values())


@app.post("/api/v1/telemetry", status_code=202, tags=["telemetry"])
async def ingest_telemetry(payload: MachineHealth):
    """
    Ingest a single telemetry record from the PySpark Structured Streaming job.

    The Spark foreachBatch sink sends a POST request to this endpoint for every
    processed micro-batch row. The data is stored in the in-memory state and
    immediately broadcast to all connected WebSocket clients.
    """
    latest_telemetry[payload.unit_number] = payload
    await broadcast(payload.model_dump())
    logger.info(
        "Ingested | unit=%d cycle=%d RUL=%.1f anomaly=%s",
        payload.unit_number, payload.cycle, payload.predicted_rul, payload.is_anomaly,
    )
    return {"status": "accepted", "unit_number": payload.unit_number}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint consumed by the React operator dashboard.

    On connect, immediately sends the full current fleet snapshot so the UI
    renders without waiting for the next Spark micro-batch.
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("WebSocket client connected. Total: %d", len(active_connections))

    # Send current state immediately so the UI is not blank on first load
    if latest_telemetry:
        await websocket.send_json({
            "type": "snapshot",
            "data": [v.model_dump() for v in latest_telemetry.values()],
        })

    try:
        # Keep the connection alive; data is pushed via broadcast()
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(active_connections))


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
