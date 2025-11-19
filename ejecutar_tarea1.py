"""
Script para ejecutar la Tarea 1 de TensorFlow de forma interactiva.
Este script ejecuta todo el flujo de entrenamiento y genera visualizaciones.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime

# Importar el modelo
from modelo_cuadratico import ModeloCuadratico

def main():
    print("\n" + "="*70)
    print(" "*15 + "TAREA 1: RED NEURONAL PARA y = x²")
    print("="*70 + "\n")
    
    # ========== PASO 0: GENERAR DATOS ==========
    print("PASO 0: Generando datos de entrenamiento...")
    modelo_cuad = ModeloCuadratico()
    x_data, y_data = modelo_cuad.generar_datos(
        n_samples=1000,
        rango=(-1, 1),
        ruido=0.02,
        seed=42
    )
    
    print(f"✓ Datos generados:")
    print(f"  - x shape: {x_data.shape}")
    print(f"  - y shape: {y_data.shape}")
    print(f"  - x rango: [{x_data.min():.3f}, {x_data.max():.3f}]")
    print(f"  - y rango: [{y_data.min():.3f}, {y_data.max():.3f}]\n")
    
    # ========== PASO 1: VISUALIZAR DATOS ==========
    print("PASO 1: Visualizando datos...")
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, alpha=0.5, s=10, c='blue', label='Datos generados (y = x² + ruido)')
    
    x_original = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_original = x_original ** 2
    plt.plot(x_original, y_original, 'r-', linewidth=2, label='Función original (y = x²)')
    
    plt.title('Conjunto de Datos Sintético', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Crear directorio de salida
    os.makedirs('outputs/tarea1', exist_ok=True)
    plt.savefig('outputs/tarea1/01_datos_generados.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: outputs/tarea1/01_datos_generados.png\n")
    
    # ========== PASO 2: CONSTRUIR MODELO ==========
    print("PASO 2: Construyendo modelo neural...")
    modelo_cuad.construir_modelo()
    
    print("✓ Modelo construido. Arquitectura:")
    modelo_cuad.modelo.summary()
    print()
    
    # ========== PASO 3: ENTRENAR MODELO ==========
    print("PASO 3: Entrenando modelo...")
    print("-" * 70)
    
    history = modelo_cuad.entrenar(
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    print("-" * 70)
    print("✓ Entrenamiento completado\n")
    
    # ========== PASO 4: VISUALIZAR CURVAS DE APRENDIZAJE ==========
    print("PASO 4: Generando curvas de aprendizaje...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Pérdida (entrenamiento)', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Pérdida (validación)', linewidth=2)
    axes[0].set_title('Evolución de la Pérdida (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='MAE (entrenamiento)', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='MAE (validación)', linewidth=2)
    axes[1].set_title('Evolución del Error Absoluto Medio (MAE)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/tarea1/02_curvas_aprendizaje.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: outputs/tarea1/02_curvas_aprendizaje.png\n")
    
    # ========== PASO 5: PREDICCIONES ==========
    print("PASO 5: Realizando predicciones...")
    predicciones = modelo_cuad.predecir(x_data)
    
    mse = mean_squared_error(y_data, predicciones)
    mae = mean_absolute_error(y_data, predicciones)
    rmse = np.sqrt(mse)
    
    print(f"✓ Métricas de Evaluación:")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - MAE: {mae:.6f}\n")
    
    # Visualizar predicciones
    indices_ordenados = np.argsort(x_data.flatten())
    x_ordenado = x_data[indices_ordenados]
    y_ordenado = y_data[indices_ordenados]
    pred_ordenado = predicciones[indices_ordenados]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(x_ordenado, y_ordenado, alpha=0.5, s=10, c='blue', label='Datos reales')
    axes[0].plot(x_ordenado, pred_ordenado, 'r-', linewidth=2, label='Predicciones del modelo')
    axes[0].set_title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    residuos = y_ordenado - pred_ordenado
    axes[1].scatter(x_ordenado, residuos, alpha=0.5, s=10, c='green')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Residuos (Errores de Predicción)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Residuo')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/tarea1/03_predicciones_residuos.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: outputs/tarea1/03_predicciones_residuos.png\n")
    
    # ========== PASO 6: PREDICCIONES ESPECÍFICAS ==========
    print("PASO 6: Predicciones en valores específicos:")
    print("-" * 60)
    
    x_prueba = np.array([[-1.0], [-0.75], [-0.5], [-0.25], [0.0], [0.25], [0.5], [0.75], [1.0]])
    y_esperado = x_prueba ** 2
    y_predicho = modelo_cuad.predecir(x_prueba)
    
    print(f"{'x':<10} {'y esperado':<15} {'y predicho':<15} {'error':<10}")
    print("-" * 60)
    
    for i, (x, y_esp, y_pred) in enumerate(zip(x_prueba, y_esperado, y_predicho)):
        error = abs(y_esp - y_pred)[0]
        print(f"{x[0]:<10.2f} {y_esp[0]:<15.6f} {y_pred[0]:<15.6f} {error:<10.6f}")
    
    print("-" * 60 + "\n")
    
    # ========== PASO 7: GUARDAR MODELO ==========
    print("PASO 7: Guardando modelo entrenado...")
    os.makedirs('modelos', exist_ok=True)
    
    modelo_cuad.guardar_modelo(
        path_tf='modelos/modelo_entrenado.h5',
        path_pkl='modelos/modelo_entrenado.pkl'
    )
    
    print()
    
    # ========== RESUMEN FINAL ==========
    print("="*70)
    print(" "*20 + "RESUMEN DEL PROYECTO")
    print("="*70)
    
    print("\n📊 DATOS:")
    print(f"  - Número de muestras: {len(x_data)}")
    print(f"  - Rango de x: [{x_data.min():.3f}, {x_data.max():.3f}]")
    print(f"  - Rango de y: [{y_data.min():.3f}, {y_data.max():.3f}]")
    
    print("\n🧠 ARQUITECTURA DEL MODELO:")
    print(f"  - Capa de entrada: 1 neurona")
    print(f"  - Capa oculta 1: 64 neuronas (ReLU)")
    print(f"  - Capa oculta 2: 64 neuronas (ReLU)")
    print(f"  - Capa de salida: 1 neurona (Linear)")
    print(f"  - Total de parámetros: {modelo_cuad.modelo.count_params():,}")
    
    print("\n📈 ENTRENAMIENTO:")
    print(f"  - Épocas realizadas: {len(history.history['loss'])}")
    print(f"  - Batch size: 32")
    print(f"  - Split de validación: 20%")
    print(f"  - Optimizador: Adam (learning_rate=0.001)")
    print(f"  - Loss: Mean Squared Error (MSE)")
    
    print("\n✅ RESULTADOS:")
    print(f"  - MSE final (entrenamiento): {history.history['loss'][-1]:.6f}")
    print(f"  - MSE final (validación): {history.history['val_loss'][-1]:.6f}")
    print(f"  - MAE final (entrenamiento): {history.history['mae'][-1]:.6f}")
    print(f"  - MAE final (validación): {history.history['val_mae'][-1]:.6f}")
    
    print("\n📊 MÉTRICAS GLOBALES:")
    print(f"  - MSE en conjunto completo: {mse:.6f}")
    print(f"  - RMSE en conjunto completo: {rmse:.6f}")
    print(f"  - MAE en conjunto completo: {mae:.6f}")
    
    print("\n💾 ARCHIVOS GENERADOS:")
    print(f"  ✓ modelos/modelo_entrenado.h5")
    print(f"  ✓ modelos/modelo_entrenado.pkl")
    print(f"  ✓ outputs/tarea1/01_datos_generados.png")
    print(f"  ✓ outputs/tarea1/02_curvas_aprendizaje.png")
    print(f"  ✓ outputs/tarea1/03_predicciones_residuos.png")
    
    # Guardar reporte JSON
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
            "capas": [
                {"nombre": "Dense 64 ReLU", "unidades": 64},
                {"nombre": "Dense 64 ReLU", "unidades": 64},
                {"nombre": "Dense 1 Linear", "unidades": 1}
            ],
            "parametros_totales": int(modelo_cuad.modelo.count_params())
        },
        "entrenamiento": {
            "epochs": len(history.history['loss']),
            "batch_size": 32,
            "validation_split": 0.2,
            "optimizador": "Adam"
        },
        "resultados": {
            "mse_final_train": float(history.history['loss'][-1]),
            "mse_final_val": float(history.history['val_loss'][-1]),
            "mae_final_train": float(history.history['mae'][-1]),
            "mae_final_val": float(history.history['val_mae'][-1]),
            "mse_conjunto_completo": float(mse),
            "rmse_conjunto_completo": float(rmse),
            "mae_conjunto_completo": float(mae)
        }
    }
    
    os.makedirs('outputs/tarea1', exist_ok=True)
    with open('outputs/tarea1/reporte.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"\n  ✓ outputs/tarea1/reporte.json")
    
    print("\n" + "="*70)
    print("¡TAREA 1 COMPLETADA CON ÉXITO!")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
