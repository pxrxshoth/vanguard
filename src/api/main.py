"""
Vanguard Dashboard API
Asynchronous FastAPI microservice to serve real-time machine health metrics and anomalies.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import asyncio
import json
import random

app = FastAPI(title="Vanguard Operator Dashboard API", version="1.0.0")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MachineHealth(BaseModel):
    unit_number: int
    cycle: int
    predicted_rul: float
    anomaly_score: float
    is_anomaly: bool

# In-memory mock data store
latest_telemetry: Dict[int, MachineHealth] = {}
active_connections: List[WebSocket] = []

@app.get("/")
async def root():
    return {"message": "Vanguard Dashboard API is running."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/v1/engines", response_model=List[MachineHealth])
async def get_engines():
    """Retrieve the latest health status of all tracked engines."""
    return list(latest_telemetry.values())

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint pushing real-time health metrics to the frontend
    under a 50ms SLA.
    """
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Simulate receiving streaming telemetry from Kafka/PySpark
            await asyncio.sleep(1.0)
            
            # Generate mock update for a random engine
            unit = random.randint(1, 100)
            rul = max(0.0, 150 - random.uniform(0, 50))
            anomaly_score = random.uniform(0, 1)
            is_anomaly = anomaly_score > 0.85
            
            payload = MachineHealth(
                unit_number=unit,
                cycle=random.randint(100, 300),
                predicted_rul=rul,
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly
            )
            
            latest_telemetry[unit] = payload
            
            # Push update to client
            await websocket.send_json(payload.dict())
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    # Serving with Uvicorn backend
    uvicorn.run(app, host="0.0.0.0", port=8000)
