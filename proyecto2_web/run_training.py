"""
Script de Entrenamiento y Demostración
=====================================

Demuestra cómo entrenar un modelo, guardarlo y luego servilo
a través del servicio web.

Flujo:
1. Generar datos sintéticos
2. Entrenar modelo neural
3. Guardar modelo y escaladores
4. Cargar en servicio web
5. Realizar predicciones de prueba
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from servicio_web import ServicioWebTensorFlow
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generar_datos(n_muestras=1000, n_features=10, random_state=42):
    """
    Genera datos sintéticos para demostración.
    
    Args:
        n_muestras: Número de muestras
        n_features: Número de características
        random_state: Seed para reproducibilidad
    
    Returns:
        Tupla (X_train, y_train)
    """
    np.random.seed(random_state)
    
    logger.info(f"📊 Generando {n_muestras} muestras con {n_features} características...")
    
    X = np.random.randn(n_muestras, n_features).astype(np.float32)
    
    # Crear etiquetas con relación no lineal
    y = (2 * X[:, 0] + 3 * X[:, 1] + 
         X[:, 2]**2 - 0.5 * X[:, 3] + 
         np.random.randn(n_muestras) * 0.1).reshape(-1, 1).astype(np.float32)
    
    logger.info(f"✅ Datos generados. X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


def entrenar_modelo(X_train, y_train, epochs=50, batch_size=32):
    """
    Entrena un modelo neural.
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        epochs: Número de épocas
        batch_size: Tamaño del lote
    
    Returns:
        Modelo entrenado
    """
    logger.info("🤖 Construyendo modelo neural...")
    
    modelo = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),
                          name='input_layer'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(32, activation='relu', name='hidden_1'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(16, activation='relu', name='hidden_2'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        
        keras.layers.Dense(1, name='output_layer')
    ])
    
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"✅ Modelo creado. Parámetros: {modelo.count_params():,}")
    logger.info(f"📚 Entrenando durante {epochs} épocas...")
    
    historial = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    logger.info("✅ Entrenamiento completado")
    
    return modelo, historial


def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo.
    
    Args:
        modelo: Modelo a evaluar
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
    
    Returns:
        Métricas de evaluación
    """
    logger.info("📊 Evaluando modelo...")
    
    y_pred = modelo.predict(X_test, verbose=0)
    
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # R²
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metricas = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    logger.info(f"✅ Métricas:")
    logger.info(f"   MSE:  {metricas['mse']:.4f}")
    logger.info(f"   RMSE: {metricas['rmse']:.4f}")
    logger.info(f"   MAE:  {metricas['mae']:.4f}")
    logger.info(f"   R²:   {metricas['r2']:.4f}")
    
    return metricas


def main():
    """Función principal."""
    
    print("\n" + "="*70)
    print("🌐 DEMOSTRACIÓN: SERVICIO WEB REST TENSORFLOW")
    print("="*70 + "\n")
    
    # 1. Generar datos
    X, y = generar_datos(n_muestras=1000, n_features=10)
    
    # 2. Dividir datos
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Normalizar datos
    logger.info("📊 Normalizando datos...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_test_scaled = scaler_X.transform(X_test)
    
    logger.info("✅ Datos normalizados")
    
    # 4. Entrenar modelo
    modelo, historial = entrenar_modelo(
        X_train_scaled, y_train_scaled,
        epochs=50,
        batch_size=32
    )
    
    # 5. Evaluar
    evaluar_modelo(modelo, X_test_scaled, y_test)
    
    # 6. Guardar modelo
    logger.info("💾 Guardando modelo...")
    Path("modelos").mkdir(exist_ok=True)
    
    servicio = ServicioWebTensorFlow(ruta_modelos="./modelos")
    servicio.guardar_modelo(
        "default",
        modelo,
        scaler_X,
        scaler_y,
        "./modelos/default"
    )
    
    logger.info("✅ Modelo guardado")
    
    # 7. Cargar en servicio
    logger.info("🔄 Cargando modelo en servicio...")
    servicio.cargar_modelo("default", "./modelos/default")
    logger.info("✅ Modelo cargado en servicio")
    
    # 8. Probar predicción
    logger.info("🎯 Realizando predicciones de prueba...")
    X_demo = X_test[:5].astype(np.float32)
    
    y_pred, confianza = servicio.predecir("default", X_demo)
    
    logger.info("✅ Predicciones realizadas:")
    for i, (pred, conf) in enumerate(zip(y_pred, confianza)):
        logger.info(f"   Muestra {i+1}: Predicción = {pred[0]:.4f}, Confianza = {conf:.4f}")
    
    # 9. Mostrar estadísticas
    logger.info("\n📈 Estadísticas del servicio:")
    stats = servicio.obtener_estadisticas()
    logger.info(f"   Modelos activos: {stats['modelos_activos']}")
    logger.info(f"   Predicciones totales: {stats['predicciones_totales']}")
    
    print("\n" + "="*70)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("="*70)
    print("\nPara iniciar el servidor web:")
    print("  uvicorn servicio_web:app --reload --host 0.0.0.0 --port 8000")
    print("\nPara acceder a la documentación:")
    print("  http://localhost:8000/docs")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
