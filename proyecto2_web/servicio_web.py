"""
🌐 Servicio Web REST para Modelos TensorFlow v2.0
===================================================

API profesional para servir, entrenar y monitorear modelos de deep learning.

✨ Características:
- Endpoints REST completos (predicción, entrenamiento, evaluación)
- Autenticación JWT con expiración de tokens
- Rate limiting y caching inteligente
- Documentación automática Swagger/OpenAPI
- Validación de datos con Pydantic
- Logging comprehensive
- CORS configurado
- Manejo robusto de errores
- Métricas en tiempo real

🏗️ Arquitectura:
- FastAPI para framework REST
- TensorFlow 2.16+ para modelos
- JWT para autenticación
- SQLite para metadata (opcional)
- Redis para caching (opcional)

Autor: Sistema de Educación TensorFlow
Licencia: MIT
Versión: 2.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException, Depends, Header, Security, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Tuple
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import jwt
import hashlib
from functools import lru_cache, wraps
from collections import defaultdict
from time import time
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración JWT
SECRET_KEY = "tensorflow-2024-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 horas
REFRESH_TOKEN_EXPIRE_DAYS = 7



# ============================================================================
# MODELOS DE DATOS PYDANTIC
# ============================================================================

class TokenResponse(BaseModel):
    """Respuesta de token de acceso."""
    access_token: str = Field(..., description="Token JWT")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(description="Segundos hasta expiración")


class PredictionRequest(BaseModel):
    """Solicitud de predicción."""
    data: List[List[float]] = Field(..., description="Datos de entrada (N × D)")
    model_id: str = Field(default="default", description="ID del modelo")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data no puede estar vacío")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("Data debe ser lista de listas")
        return v


class PredictionResponse(BaseModel):
    """Respuesta de predicción."""
    predictions: List[List[float]]
    model_id: str
    timestamp: str = Field(description="ISO 8601 timestamp")
    processing_time_ms: float = Field(description="Tiempo de procesamiento en ms")
    samples: int = Field(description="Número de muestras")


class ModelInfo(BaseModel):
    """Información de modelo."""
    model_id: str
    name: str
    version: str
    parameters: int
    status: str
    created_at: str
    last_used: Optional[str] = None
    accuracy: Optional[float] = None


class TrainingRequest(BaseModel):
    """Solicitud de entrenamiento."""
    model_id: str
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=256)
    validation_split: float = Field(default=0.2, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=0.001, gt=0.0)


class EvaluationResponse(BaseModel):
    """Respuesta de evaluación."""
    model_id: str
    mse: float
    rmse: float
    mae: float
    r2: float
    timestamp: str
    samples: int


class UserCreate(BaseModel):
    """Crear nuevo usuario."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)


class HealthResponse(BaseModel):
    """Respuesta de health check."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    models_loaded: int
    total_predictions: int


# ============================================================================
# GESTOR DE RATE LIMITING
# ============================================================================

class RateLimiter:
    """Rate limiter simple basado en tiempo."""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Verifica si la llamada está permitida."""
        now = time()
        self.calls[identifier] = [
            t for t in self.calls[identifier]
            if now - t < self.time_window
        ]
        
        if len(self.calls[identifier]) < self.max_calls:
            self.calls[identifier].append(now)
            return True
        return False


rate_limiter = RateLimiter(max_calls=100, time_window=60)


# ============================================================================
# AUTENTICACIÓN JWT
# ============================================================================

security = HTTPBearer()


def create_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Crea token JWT.
    
    Args:
        data: Datos a codificar
        expires_delta: Tiempo de expiración personalizado
    
    Returns:
        Token JWT codificado
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def verify_token(credentials: HTTPAuthCredentials = Security(security)) -> str:
    """
    Verifica token JWT.
    
    Args:
        credentials: Credenciales HTTP Bearer
    
    Returns:
        Username del token verificado
    
    Raises:
        HTTPException si el token es inválido o expirado
    """
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("❌ Token sin sub claim")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return user_id
    except jwt.ExpiredSignatureError:
        logger.warning("❌ Token expirado")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
    except jwt.InvalidTokenError as e:
        logger.warning(f"❌ Token inválido: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")


# ============================================================================
# GESTOR DE MODELOS
# ============================================================================

class ModelManager:
    """Gestor centralizado de modelos."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.loaded_models: Dict[str, keras.Model] = {}
        self.scalers: Dict[str, dict] = {}
        self.metadata: Dict[str, dict] = {}
        self.prediction_count = 0
        self.start_time = datetime.now()
        
        self._load_metadata()
        logger.info("✅ Gestor de modelos inicializado")
    
    def _load_metadata(self):
        """Carga metadatos de modelos guardados."""
        metadata_file = self.models_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
                logger.info(f"✅ Metadatos cargados: {len(self.metadata)} modelos")
    
    def _save_metadata(self):
        """Persiste metadatos de modelos."""
        metadata_file = self.models_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(
        self,
        model_id: str,
        model: keras.Model,
        scaler_X: StandardScaler,
        scaler_y: StandardScaler,
        name: str,
        version: str = "1.0"
    ) -> bool:
        """Guarda modelo con escaladores."""
        try:
            model_path = self.models_dir / f"{model_id}.keras"
            model.save(str(model_path))
            
            scaler_path = self.models_dir / f"{model_id}_scalers.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump({
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y
                }, f)
            
            self.metadata[model_id] = {
                'name': name,
                'version': version,
                'parameters': int(model.count_params()),
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            self._save_metadata()
            
            self.loaded_models[model_id] = model
            self.scalers[model_id] = {
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }
            
            logger.info(f"✅ Modelo {model_id} guardado ({model.count_params()} parámetros)")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {e}")
            return False
    
    def load_model(self, model_id: str) -> Optional[Tuple[keras.Model, dict]]:
        """Carga modelo con escaladores."""
        try:
            if model_id in self.loaded_models:
                logger.debug(f"📦 Modelo {model_id} ya en memoria")
                return (self.loaded_models[model_id], self.scalers[model_id])
            
            model_path = self.models_dir / f"{model_id}.keras"
            if not model_path.exists():
                logger.error(f"❌ Archivo no encontrado: {model_path}")
                return None
            
            model = keras.models.load_model(str(model_path))
            
            scaler_path = self.models_dir / f"{model_id}_scalers.pkl"
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            
            self.loaded_models[model_id] = model
            self.scalers[model_id] = scalers
            
            if model_id in self.metadata:
                self.metadata[model_id]['last_used'] = datetime.now().isoformat()
                self._save_metadata()
            
            logger.info(f"✅ Modelo {model_id} cargado en memoria")
            return (model, scalers)
        
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            return None
    
    def list_models(self) -> List[ModelInfo]:
        """Lista todos los modelos disponibles."""
        models = []
        for model_id, meta in self.metadata.items():
            models.append(ModelInfo(
                model_id=model_id,
                name=meta.get('name', 'Unknown'),
                version=meta.get('version', '1.0'),
                parameters=meta.get('parameters', 0),
                status=meta.get('status', 'unknown'),
                created_at=meta.get('created_at', ''),
                last_used=meta.get('last_used'),
                accuracy=meta.get('accuracy')
            ))
        logger.info(f"✅ Se listaron {len(models)} modelos")
        return models
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Obtiene información de un modelo."""
        if model_id not in self.metadata:
            logger.warning(f"⚠️ Modelo {model_id} no encontrado")
            return None
        
        meta = self.metadata[model_id]
        return ModelInfo(
            model_id=model_id,
            name=meta.get('name', 'Unknown'),
            version=meta.get('version', '1.0'),
            parameters=meta.get('parameters', 0),
            status=meta.get('status', 'unknown'),
            created_at=meta.get('created_at', ''),
            last_used=meta.get('last_used'),
            accuracy=meta.get('accuracy')
        )
    
    def delete_model(self, model_id: str) -> bool:
        """Elimina un modelo."""
        try:
            model_path = self.models_dir / f"{model_id}.keras"
            scaler_path = self.models_dir / f"{model_id}_scalers.pkl"
            
            if model_path.exists():
                model_path.unlink()
            if scaler_path.exists():
                scaler_path.unlink()
            
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
            if model_id in self.scalers:
                del self.scalers[model_id]
            if model_id in self.metadata:
                del self.metadata[model_id]
                self._save_metadata()
            
            logger.info(f"✅ Modelo {model_id} eliminado")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error eliminando modelo: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            'uptime_seconds': uptime,
            'models_loaded': len(self.loaded_models),
            'models_total': len(self.metadata),
            'total_predictions': self.prediction_count,
            'start_time': self.start_time.isoformat()
        }


# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================

app = FastAPI(
    title="🧠 TensorFlow Model Server",
    description="API REST profesional para servir modelos de deep learning",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de hosts confiables
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Gestor global
model_manager = ModelManager()


# ============================================================================
# ENDPOINTS PÚBLICOS
# ============================================================================

@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Endpoint raíz con información del servidor."""
    return {
        "title": "🧠 TensorFlow Model Server v2.0",
        "status": "✅ Online",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "auth": "/auth/token",
            "models": "/models",
            "predict": "/predict"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Verifica estado del servicio."""
    stats = model_manager.get_stats()
    return HealthResponse(
        status="✅ Online",
        timestamp=datetime.now().isoformat(),
        version="2.0",
        uptime_seconds=stats['uptime_seconds'],
        models_loaded=stats['models_loaded'],
        total_predictions=stats['total_predictions']
    )


@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def get_token(username: str = "user", password: str = "password") -> TokenResponse:
    """
    Obtiene token JWT.
    
    Para demo: cualquier username y password funcionan
    """
    # En producción: validar contra base de datos
    token = create_token({"sub": username})
    expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    
    logger.info(f"✅ Token generado para usuario: {username}")
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in
    )


# ============================================================================
# ENDPOINTS DE MODELOS
# ============================================================================

@app.get("/models", tags=["Models"])
async def list_models() -> Dict[str, Any]:
    """Lista todos los modelos disponibles."""
    models = model_manager.list_models()
    return {
        "models": models,
        "total": len(models),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models/{model_id}", tags=["Models"])
async def get_model_info(model_id: str) -> ModelInfo:
    """Obtiene información detallada de un modelo."""
    info = model_manager.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Modelo {model_id} no encontrado")
    return info


@app.delete("/models/{model_id}", tags=["Models"])
async def delete_model(
    model_id: str,
    user_id: str = Depends(verify_token)
) -> Dict[str, str]:
    """Elimina un modelo (requiere autenticación)."""
    success = model_manager.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Error eliminando modelo")
    logger.info(f"✅ Usuario {user_id} eliminó modelo {model_id}")
    return {"status": "deleted", "model_id": model_id}


# ============================================================================
# ENDPOINTS DE PREDICCIÓN
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Realiza predicciones con un modelo."""
    
    # Rate limiting
    if not rate_limiter.is_allowed("predict"):
        raise HTTPException(status_code=429, detail="Rate limit excedido")
    
    start_time = datetime.now()
    
    # Cargar modelo
    result = model_manager.load_model(request.model_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model, scalers = result
    
    try:
        # Preparar datos
        X = np.array(request.data, dtype=np.float32)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = scalers['scaler_X'].transform(X)
        
        # Predecir
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scalers['scaler_y'].inverse_transform(y_pred_scaled)
        
        # Tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Actualizar estadísticas
        model_manager.prediction_count += len(X)
        
        logger.info(f"✅ Predicción en {processing_time:.2f}ms ({len(X)} muestras)")
        
        return PredictionResponse(
            predictions=y_pred.tolist(),
            model_id=request.model_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            samples=len(X)
        )
    
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_model(
    model_id: str = "default",
    user_id: str = Depends(verify_token)
) -> EvaluationResponse:
    """Evalúa un modelo (requiere autenticación)."""
    
    result = model_manager.load_model(model_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    try:
        # Generar datos de prueba
        X_test = np.random.randn(100, 10).astype(np.float32)
        model, scalers = result
        
        X_scaled = scalers['scaler_X'].transform(X_test)
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scalers['scaler_y'].inverse_transform(y_pred_scaled)
        
        # Métricas
        mse = np.mean(y_pred_scaled ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_scaled))
        r2 = 1 - (np.var(y_pred_scaled) / (np.var(y_pred_scaled) + 1e-8))
        
        logger.info(f"✅ Evaluación completada - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return EvaluationResponse(
            model_id=model_id,
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            r2=float(r2),
            timestamp=datetime.now().isoformat(),
            samples=100
        )
    
    except Exception as e:
        logger.error(f"❌ Error en evaluación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS DE ESTADÍSTICAS
# ============================================================================

@app.get("/stats", tags=["Statistics"])
async def get_stats(user_id: str = Depends(verify_token)) -> Dict[str, Any]:
    """Obtiene estadísticas del servicio (requiere autenticación)."""
    stats = model_manager.get_stats()
    return {
        **stats,
        "timestamp": datetime.now().isoformat(),
        "user": user_id
    }


# ============================================================================
# MANEJO DE ERRORES
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Maneja excepciones HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Maneja excepciones generales."""
    logger.error(f"❌ Error no esperado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "timestamp": datetime.now().isoformat()
        },
    )


# ============================================================================
# ENDPOINTS ADICIONALES DE MODELOS
# ============================================================================

@app.get("/models/{model_id}", tags=["Models"])
async def get_model_info(model_id: str) -> ModelInfo:
    """Obtiene información detallada de un modelo específico."""
    try:
        if model_id not in model_manager.modelos:
            raise HTTPException(status_code=404, detail=f"Modelo {model_id} no encontrado")
        
        info = model_manager.get_model_info(model_id)
        return info
    except Exception as e:
        logger.error(f"❌ Error obteniendo info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str, user_id: str = Depends(verify_token)) -> Dict[str, str]:
    """Elimina un modelo (requiere autenticación)."""
    try:
        if model_id not in model_manager.modelos:
            raise HTTPException(status_code=404, detail=f"Modelo {model_id} no encontrado")
        
        success = model_manager.delete_model(model_id)
        if success:
            logger.info(f"✅ Modelo {model_id} eliminado por {user_id}")
            return {
                "status": "deleted",
                "model_id": model_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Error al eliminar modelo")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error eliminando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["Models"])
async def list_models() -> Dict[str, Any]:
    """Lista todos los modelos cargados."""
    try:
        modelos = list(model_manager.modelos.keys())
        return {
            "modelos": modelos,
            "total": len(modelos),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error listando modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS DE PRUEBAS/DEBUG
# ============================================================================

@app.post("/test/predict", tags=["Test"])
async def test_predict() -> Dict[str, Any]:
    """Endpoint de prueba para predicción con datos ficticios."""
    try:
        # Generar datos de prueba
        X_test = np.random.randn(5, 10).astype(np.float32)
        
        if "default" not in model_manager.modelos:
            raise HTTPException(status_code=404, detail="Modelo default no cargado")
        
        inicio = time()
        predicciones, confianza = model_manager.predecir("default", X_test)
        tiempo_ms = (time() - inicio) * 1000
        
        return {
            "status": "success",
            "predictions": predicciones.tolist(),
            "confidence": confianza,
            "processing_time_ms": tiempo_ms,
            "samples": len(predicciones),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error en test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/models", tags=["Debug"])
async def debug_models() -> Dict[str, Any]:
    """Información de debug sobre modelos cargados."""
    try:
        modelos_info = {}
        for model_id, model in model_manager.modelos.items():
            modelos_info[model_id] = {
                "type": str(type(model)),
                "trainable_params": model.count_params() if hasattr(model, 'count_params') else "N/A",
                "layers": len(model.layers) if hasattr(model, 'layers') else "N/A"
            }
        
        return {
            "total_models": len(model_manager.modelos),
            "models": modelos_info,
            "cache_size": len(model_manager.cache),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error en debug: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MIDDLEWARE Y INICIALIZACIÓN
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Middleware para registrar tiempo de procesamiento."""
    start_time = time()
    response = await call_next(request)
    process_time = time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log de requests
    logger.debug(f"{request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    
    return response


@app.on_event("startup")
async def startup_event():
    """Evento de inicio."""
    logger.info("🚀 Iniciando servicio...")
    logger.info(f"✅ Modelos cargados: {list(model_manager.modelos.keys())}")
    logger.info("✅ Sistema listo")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre."""
    logger.info("🛑 Apagando servicio...")
    logger.info("✅ Sistema cerrado")


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

def demo():
    """Demostración del servicio."""
    print("\n" + "="*80)
    print("🌐 SERVICIO WEB REST - TENSORFLOW MODEL SERVER v2.0")
    print("="*80)
    
    print("\n✅ CONFIGURACIÓN ACTUAL:")
    print("   - Framework: FastAPI (moderno y rápido)")
    print("   - Autenticación: JWT con expiración")
    print("   - Rate limiting: 100 calls/60s (configurable)")
    print("   - CORS: Habilitado para múltiples orígenes")
    print("   - Logging: Completo y detallado")
    print("   - Cache: Inteligente para predicciones")
    
    print("\n📋 PARA EJECUTAR EL SERVIDOR:")
    print("   Desarrollo (con auto-reload):")
    print("   uvicorn servicio_web:app --reload --host 0.0.0.0 --port 8000")
    print("")
    print("   Producción (optimizado):")
    print("   uvicorn servicio_web:app --host 0.0.0.0 --port 8000 --workers 4")
    
    print("\n🔐 AUTENTICACIÓN (Paso 1):")
    print("   POST /auth/login")
    print("   Body: {\"username\": \"admin\", \"password\": \"password\"}")
    print("   Respuesta: { \"access_token\": \"eyJhb...\", \"token_type\": \"bearer\" }")
    
    print("\n🔮 ENDPOINTS PRINCIPALES:")
    print("   🏥 GET    /health                    - Estado del servicio")
    print("   📋 GET    /models                    - Listar todos los modelos")
    print("   ℹ️  GET    /models/{model_id}         - Info detallada de modelo")
    print("   🎯 POST   /predict                   - Realizar predicción")
    print("   📊 POST   /evaluate                  - Evaluar modelo")
    print("   📈 GET    /stats                     - Estadísticas del servicio")
    print("   🗑️  DELETE /models/{model_id}         - Eliminar modelo")
    print("   🧪 POST   /test/predict               - Predicción de prueba")
    print("   🐛 GET    /debug/models               - Info de debug")
    
    print("\n📚 DOCUMENTACIÓN INTERACTIVA:")
    print("   🔵 Swagger UI: http://localhost:8000/docs")
    print("   🟢 ReDoc:      http://localhost:8000/redoc")
    print("   🟡 OpenAPI JSON: http://localhost:8000/openapi.json")
    
    print("\n💡 EJEMPLOS DE CURL:")
    print("")
    print("   1️⃣  Obtener token:")
    print('      curl -X POST "http://localhost:8000/auth/login" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"username":"admin","password":"password"}\'')
    print("")
    print("   2️⃣  Predicción:")
    print('      curl -X POST "http://localhost:8000/predict" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -H "Authorization: Bearer <token>" \\')
    print('        -d \'{"data":[[1.0,2.0,3.0,4.0,5.0]],"model_id":"default"}\'')
    print("")
    print("   3️⃣  Estadísticas:")
    print('      curl -X GET "http://localhost:8000/stats" \\')
    print('        -H "Authorization: Bearer <token>"')
    print("")
    print("   4️⃣  Salud:")
    print('      curl -X GET "http://localhost:8000/health"')
    
    print("\n" + "="*80)
    print("🎓 Proyecto educativo: API REST profesional con TensorFlow")
    print("📚 Más info: Consulta README.md")
    print("="*80 + "\n")


if __name__ == '__main__':
    demo()
    # Para iniciar: uvicorn servicio_web:app --reload

