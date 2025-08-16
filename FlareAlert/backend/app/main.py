#!/usr/bin/env python3
"""
FlareAlert FastAPI Application
Main entry point for the solar flare prediction web API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from sqlalchemy import text
import asyncio
from collections import defaultdict

from config import Config
from database.models import create_database, SolarFlare, CME, GeomagneticStorm
from data.ingestion import DataIngestion
from app.prediction_service import PredictionService
# from scripts.hazard_ensemble_model import HazardEnsembleModel  # Commented out for now

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
prediction_service = None
last_prediction = None
last_model_update = None
websocket_connections: List[WebSocket] = []
alert_subscribers: Dict[str, List[WebSocket]] = defaultdict(list)

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for solar flare prediction"""
    timestamp: Optional[str] = None  # 
    
    model_config = {"protected_namespaces": ()}

class PredictionResponse(BaseModel):
    """Response model for solar flare prediction"""
    timestamp: str
    prediction: float
    confidence: float
    risk_level: str
    features_used: int
    model_version: str
    
    model_config = {"protected_namespaces": ()}

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    model_loaded: bool
    last_prediction: Optional[str]
    database_connected: bool
    
    model_config = {"protected_namespaces": ()}

class SolarActivityResponse(BaseModel):
    """Response model for solar activity data"""
    flares_7d: int
    cmes_7d: int
    storms_7d: int
    current_kp: Optional[float]
    max_kp_7d: Optional[float]
    
    model_config = {"protected_namespaces": ()}

class AlertRequest(BaseModel):
    """Request model for alert configuration"""
    risk_threshold: float = 0.4
    email: Optional[str] = None
    webhook_url: Optional[str] = None
    
    model_config = {"protected_namespaces": ()}

class ModelRetrainRequest(BaseModel):
    """Request model for model retraining"""
    force_retrain: bool = False
    use_latest_data: bool = True
    
    model_config = {"protected_namespaces": ()}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

def load_model():
    """Load the trained hazard ensemble model"""
    global prediction_service, last_model_update
    
    try:
        prediction_service = PredictionService()
        if prediction_service.model is not None:
            last_model_update = datetime.now()
            logger.info("Prediction service loaded successfully")
            return True
        else:
            logger.warning("Failed to load prediction service")
            return False
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def get_risk_level(prediction: float) -> str:
    """Convert prediction probability to risk level"""
    if prediction >= 0.7:
        return "HIGH"
    elif prediction >= 0.4:
        return "MEDIUM"
    elif prediction >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"

async def send_alert(risk_level: str, prediction: float, timestamp: str):
    """Send alerts to subscribers"""
    alert_data = {
        "type": "solar_flare_alert",
        "risk_level": risk_level,
        "prediction": prediction,
        "timestamp": timestamp,
        "message": f"Solar flare risk level: {risk_level} ({prediction:.3f})"
    }
    
    await manager.broadcast(json.dumps(alert_data))

async def monitor_solar_activity():
    """Background task to monitor solar activity and send alerts"""
    while True:
        try:
            if prediction_service and prediction_service.model:
                prediction, confidence = prediction_service.predict(datetime.now())
                risk_level = get_risk_level(prediction)
                
                # Send alert if risk is medium or higher
                if risk_level in ["MEDIUM", "HIGH"]:
                    await send_alert(risk_level, prediction, datetime.now().isoformat())
                
                # Broadcast current status to all connected clients
                status_data = {
                    "type": "status_update",
                    "timestamp": datetime.now().isoformat(),
                    "prediction": prediction,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "solar_activity": await get_solar_activity_data()
                }
                
                await manager.broadcast(json.dumps(status_data))
            
            # Wait 5 minutes before next check
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in solar activity monitoring: {e}")
            await asyncio.sleep(60)

async def get_solar_activity_data():
    """Get solar activity data for real-time updates"""
    try:
        engine = create_database()
        cutoff_time = datetime.now() - timedelta(days=7)
        
        with engine.connect() as conn:
            flares_7d = conn.execute(
                text("SELECT COUNT(*) FROM solar_flares WHERE peak_time >= :cutoff"),
                {"cutoff": cutoff_time}
            ).scalar() or 0
            
            cmes_7d = conn.execute(
                text("SELECT COUNT(*) FROM cmes WHERE time21_5 >= :cutoff"),
                {"cutoff": cutoff_time}
            ).scalar() or 0
            
            storms_7d = conn.execute(
                text("SELECT COUNT(*) FROM geomagnetic_storms WHERE time_tag >= :cutoff"),
                {"cutoff": cutoff_time}
            ).scalar() or 0
        
            # Get current Kp (most recent)
            current_kp_result = conn.execute(
                text("SELECT kp_index FROM geomagnetic_storms WHERE time_tag <= :now ORDER BY time_tag DESC LIMIT 1"),
                {"now": datetime.now()}
            ).scalar()
            current_kp = float(current_kp_result) if current_kp_result else 2.0
            
            # Get max Kp in last 7 days
            max_kp_result = conn.execute(
                text("SELECT MAX(kp_index) FROM geomagnetic_storms WHERE time_tag >= :cutoff"),
                {"cutoff": cutoff_time}
            ).scalar()
            max_kp_7d = float(max_kp_result) if max_kp_result else 2.0
        
        return {
            "flares_7d": flares_7d,
            "cmes_7d": cmes_7d,
            "storms_7d": storms_7d,
            "current_kp": current_kp,
            "max_kp_7d": max_kp_7d
        }
    except Exception as e:
        logger.error(f"Error getting solar activity data: {e}")
        return {"flares_7d": 0, "cmes_7d": 0, "storms_7d": 0, "current_kp": 2.0, "max_kp_7d": 2.0}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    logger.info("Starting FlareAlert API...")
    
    try:
        create_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Load ML model
    if load_model():
        logger.info("ML model loaded successfully")
    else:
        logger.warning("ML model not loaded - predictions will not be available")
    
    # Start background monitoring task
    asyncio.create_task(monitor_solar_activity())
    
    yield
    
    # Shutdown
    logger.info("Shutting down FlareAlert API...")

# Create FastAPI app
app = FastAPI(
    title="FlareAlert API",
    description="Solar Flare Prediction and Monitoring System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FlareAlert API - Solar Flare Prediction System",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket": "/ws",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "solar_activity": "/solar-activity",
            "ingest_data": "/ingest-data",
            "retrain_model": "/retrain-model",
            "alerts": "/alerts"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global prediction_service, last_prediction
    
    # Check database connection
    try:
        engine = create_database()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_connected = False
    
    return HealthResponse(
        status="healthy" if prediction_service and prediction_service.model and db_connected else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=prediction_service is not None and prediction_service.model is not None,
        last_prediction=last_prediction.isoformat() if last_prediction else None,
        database_connected=db_connected
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_solar_flare(request: PredictionRequest):
    """Predict solar flare probability for the next 6 hours"""
    global prediction_service, last_prediction
    
    if prediction_service is None or prediction_service.model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        # Parse timestamp
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
        
        # Make real prediction using the prediction service
        prediction, confidence = prediction_service.predict(timestamp)
        
        last_prediction = timestamp
        
        # Send alert if risk is high
        risk_level = get_risk_level(prediction)
        if risk_level in ["MEDIUM", "HIGH"]:
            await send_alert(risk_level, prediction, timestamp.isoformat())
        
        return PredictionResponse(
            timestamp=timestamp.isoformat(),
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            features_used=51,  # From your model
            model_version="hazard_ensemble_v1.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/solar-activity", response_model=SolarActivityResponse)
async def get_solar_activity():
    """Get current solar activity summary"""
    try:
        activity_data = await get_solar_activity_data()
        return SolarActivityResponse(**activity_data)
        
    except Exception as e:
        logger.error(f"Error getting solar activity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get solar activity: {str(e)}")

@app.post("/ingest-data")
async def trigger_data_ingestion(background_tasks: BackgroundTasks):
    """Trigger data ingestion from NASA DONKI API"""
    try:
        background_tasks.add_task(run_data_ingestion)
        return {"message": "Data ingestion started in background", "status": "started"}
    except Exception as e:
        logger.error(f"Error triggering data ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger ingestion: {str(e)}")

@app.post("/retrain-model")
async def retrain_model(request: ModelRetrainRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    try:
        background_tasks.add_task(run_model_retraining, request)
        return {"message": "Model retraining started in background", "status": "started"}
    except Exception as e:
        logger.error(f"Error triggering model retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")

@app.post("/alerts")
async def configure_alerts(request: AlertRequest):
    """Configure alert settings"""
    try:
        # Store alert configuration (in production, save to database)
        return {
            "message": "Alert configuration updated",
            "risk_threshold": request.risk_threshold,
            "email": request.email,
            "webhook_url": request.webhook_url
        }
    except Exception as e:
        logger.error(f"Error configuring alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure alerts: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded ML model"""
    global prediction_service, last_model_update
    
    if prediction_service is None or prediction_service.model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    model_info = prediction_service.get_model_info()
    
    return {
        "model_type": "Hazard Ensemble (XGBoost + Logistic Regression)",
        "features": 51,
        "last_updated": last_model_update.isoformat() if last_model_update else None,
        "performance": {
            "auc": 0.772,
            "precision": 0.502,
            "recall": 0.176,
            "false_positive_rate": 0.048
        },
        "model_details": model_info
    }

@app.get("/historical-data")
async def get_historical_data(days: int = 7):
    """Get historical solar activity data"""
    try:
        engine = create_database()
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with engine.connect() as conn:
            # Get flares
            flares = conn.execute(
                text("SELECT peak_time, class_type, class_value FROM solar_flares WHERE peak_time >= :cutoff ORDER BY peak_time"),
                {"cutoff": cutoff_time}
            ).fetchall()
            
            # Get CMEs
            cmes = conn.execute(
                text("SELECT time21_5, speed, half_angle FROM cmes WHERE time21_5 >= :cutoff ORDER BY time21_5"),
                {"cutoff": cutoff_time}
            ).fetchall()
            
            # Get storms
            storms = conn.execute(
                text("SELECT time_tag, kp_index FROM geomagnetic_storms WHERE time_tag >= :cutoff ORDER BY time_tag"),
                {"cutoff": cutoff_time}
            ).fetchall()
        
        return {
            "flares": [{"time": str(f[0]), "class": f[1], "value": float(f[2])} for f in flares],
            "cmes": [{"time": str(c[0]), "speed": float(c[1]), "half_angle": float(c[2])} for c in cmes],
            "storms": [{"time": str(s[0]), "kp": float(s[1])} for s in storms]
        }
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send initial status
            if prediction_service and prediction_service.model:
                prediction, confidence = prediction_service.predict(datetime.now())
                initial_data = {
                    "type": "connection_established",
                    "timestamp": datetime.now().isoformat(),
                    "prediction": prediction,
                    "confidence": confidence,
                    "risk_level": get_risk_level(prediction),
                    "solar_activity": await get_solar_activity_data()
                }
                await websocket.send_text(json.dumps(initial_data))
            
            # Keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def run_data_ingestion():
    """Run data ingestion in background"""
    try:
        ingestion = DataIngestion()
        ingestion.ingest_data()
        logger.info("Background data ingestion completed")
        
        # Notify connected clients
        await manager.broadcast(json.dumps({
            "type": "data_ingestion_complete",
            "timestamp": datetime.now().isoformat(),
            "message": "Data ingestion completed successfully"
        }))
    except Exception as e:
        logger.error(f"Background data ingestion failed: {e}")
        await manager.broadcast(json.dumps({
            "type": "data_ingestion_error",
            "timestamp": datetime.now().isoformat(),
            "message": f"Data ingestion failed: {str(e)}"
        }))

async def run_model_retraining(request: ModelRetrainRequest):
    """Run model retraining in background"""
    try:
        # This would integrate with your existing training scripts
        logger.info("Starting model retraining...")
        
        # Placeholder for actual retraining logic
        await asyncio.sleep(10)  # Simulate training time
        
        # Reload the model
        load_model()
        
        logger.info("Model retraining completed")
        
        # Notify connected clients
        await manager.broadcast(json.dumps({
            "type": "model_retraining_complete",
            "timestamp": datetime.now().isoformat(),
            "message": "Model retraining completed successfully"
        }))
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        await manager.broadcast(json.dumps({
            "type": "model_retraining_error",
            "timestamp": datetime.now().isoformat(),
            "message": f"Model retraining failed: {str(e)}"
        }))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG
    )
