"""
Ejecutor de Tarea 1 - Versión simplificada sin visualizaciones interactivas
Ejecuta el entrenamiento y guarda los resultados en archivos
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TensorFlow

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import json
    from datetime import datetime
    
    print("\n" + "="*70)
    print(" "*15 + "TAREA 1: RED NEURONAL PARA y = x²")
    print("="*70 + "\n")
    
    # ========== GENERAR DATOS ==========
    print("PASO 0: Generando datos...")
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, (1000, 1)).astype(np.float32)
    y_data = (x_data ** 2 + np.random.normal(0, 0.02, (1000, 1))).astype(np.float32)
    
    print(f"✓ Datos generados: {x_data.shape} muestras")
    
    # ========== CONSTRUIR MODELO ==========
    print("\nPASO 1: Construyendo modelo neural...")
    modelo = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,), name='capa_oculta_1'),
        layers.Dense(64, activation='relu', name='capa_oculta_2'),
        layers.Dense(1, activation='linear', name='capa_salida')
    ], name='ModeloCuadratico')
    
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"✓ Modelo construido con {modelo.count_params():,} parámetros")
    
    # ========== ENTRENAR ==========
    print("\nPASO 2: Entrenando modelo...")
    print("-" * 70)
    
    history = modelo.fit(
        x_data, y_data,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ModelCheckpoint(filepath='modelo_temp.h5', monitor='val_loss', save_best_only=True, verbose=0)
        ],
        verbose=1
    )
    
    print("-" * 70)
    print("✓ Entrenamiento completado")
    
    # ========== EVALUAR ==========
    print("\nPASO 3: Evaluando modelo...")
    predicciones = modelo.predict(x_data, verbose=0)
    
    mse = mean_squared_error(y_data, predicciones)
    mae = mean_absolute_error(y_data, predicciones)
    rmse = np.sqrt(mse)
    
    print(f"✓ MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # ========== GUARDAR MODELO ==========
    print("\nPASO 4: Guardando modelo...")
    os.makedirs('modelos', exist_ok=True)
    
    modelo.save('modelos/modelo_entrenado.h5')
    print("✓ Modelo guardado en: modelos/modelo_entrenado.h5")
    
    # ========== PREDICCIONES ESPECÍFICAS ==========
    print("\nPASO 5: Predicciones en valores específicos:")
    print("-" * 60)
    
    x_prueba = np.array([[-1.0], [-0.75], [-0.5], [-0.25], [0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)
    y_esperado = x_prueba ** 2
    y_predicho = modelo.predict(x_prueba, verbose=0)
    
    print(f"{'x':<10} {'y esperado':<15} {'y predicho':<15} {'error':<10}")
    print("-" * 60)
    
    for x, y_esp, y_pred in zip(x_prueba, y_esperado, y_predicho):
        error = abs(y_esp - y_pred)[0]
        print(f"{x[0]:<10.2f} {y_esp[0]:<15.6f} {y_pred[0]:<15.6f} {error:<10.6f}")
    
    print("-" * 60)
    
    # ========== REPORTE FINAL ==========
    print("\n" + "="*70)
    print(" "*20 + "RESUMEN DEL PROYECTO")
    print("="*70)
    
    print("\n📊 DATOS:")
    print(f"  - Número de muestras: {len(x_data)}")
    print(f"  - Rango de x: [{x_data.min():.3f}, {x_data.max():.3f}]")
    print(f"  - Rango de y: [{y_data.min():.3f}, {y_data.max():.3f}]")
    
    print("\n🧠 ARQUITECTURA:")
    print(f"  - Capa entrada: 1 neurona")
    print(f"  - Capa oculta 1: 64 neuronas (ReLU)")
    print(f"  - Capa oculta 2: 64 neuronas (ReLU)")
    print(f"  - Capa salida: 1 neurona (Linear)")
    print(f"  - Total parámetros: {modelo.count_params():,}")
    
    print("\n📈 ENTRENAMIENTO:")
    print(f"  - Épocas: {len(history.history['loss'])}")
    print(f"  - Batch size: 32")
    print(f"  - Validación: 20%")
    print(f"  - MSE final (train): {history.history['loss'][-1]:.6f}")
    print(f"  - MSE final (val): {history.history['val_loss'][-1]:.6f}")
    print(f"  - MAE final (train): {history.history['mae'][-1]:.6f}")
    print(f"  - MAE final (val): {history.history['val_mae'][-1]:.6f}")
    
    print("\n✅ MÉTRICAS FINALES:")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - MAE: {mae:.6f}")
    
    # Guardar reporte
    os.makedirs('outputs/tarea1', exist_ok=True)
    reporte = {
        "titulo": "Tarea 1: Red Neuronal para y = x²",
        "fecha": datetime.now().isoformat(),
        "datos": {
            "n_samples": len(x_data),
            "x_rango": [float(x_data.min()), float(x_data.max())],
            "y_rango": [float(y_data.min()), float(y_data.max())]
        },
        "modelo": {
            "arquitectura": "Sequential",
            "parametros": int(modelo.count_params())
        },
        "resultados": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mse_train": float(history.history['loss'][-1]),
            "mse_val": float(history.history['val_loss'][-1]),
            "mae_train": float(history.history['mae'][-1]),
            "mae_val": float(history.history['val_mae'][-1])
        }
    }
    
    with open('outputs/tarea1/reporte.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print("\n💾 ARCHIVOS GUARDADOS:")
    print("  ✓ modelos/modelo_entrenado.h5")
    print("  ✓ outputs/tarea1/reporte.json")
    
    print("\n" + "="*70)
    print("¡TAREA 1 COMPLETADA CON ÉXITO!")
    print("="*70 + "\n")
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
