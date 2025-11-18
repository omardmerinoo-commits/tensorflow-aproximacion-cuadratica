#!/usr/bin/env python3
"""
Script principal para entrenar el modelo de aproximación cuadrática (versión corregida).
Evita problemas con matplotlib en Jupyter.
"""

import os
# Configurar matplotlib backend ANTES de importar tensorflow/keras que carga matplotlib
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
from datetime import datetime

# Importar la clase ModeloCuadratico
from modelo_cuadratico import ModeloCuadratico


def configurar_semillas(seed: int = 42) -> None:
    """
    Configura las semillas para reproducibilidad.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    print(f"✓ Semillas configuradas (seed={seed}) para reproducibilidad\n")


def crear_directorio_resultados() -> str:
    """
    Crea un directorio para guardar los resultados del entrenamiento.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_resultados = f"resultados_{timestamp}"
    
    if not os.path.exists(dir_resultados):
        os.makedirs(dir_resultados)
        print(f"✓ Directorio de resultados creado: {dir_resultados}\n")
    
    return dir_resultados


def graficar_predicciones(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    ruta_guardado: str = "prediccion_vs_real.png"
) -> None:
    """
    Genera y guarda una gráfica comparando predicciones vs valores reales.
    """
    plt.figure(figsize=(12, 6))
    
    # Ordenar datos por x para mejor visualización
    indices_ordenados = np.argsort(x_test.flatten())
    x_sorted = x_test[indices_ordenados]
    y_test_sorted = y_test[indices_ordenados]
    y_pred_sorted = y_pred[indices_ordenados]
    
    # Gráfica 1: Comparación directa
    plt.subplot(1, 2, 1)
    plt.scatter(x_sorted, y_test_sorted, alpha=0.5, s=20, label='Datos reales', color='blue')
    plt.scatter(x_sorted, y_pred_sorted, alpha=0.5, s=20, label='Predicciones', color='red')
    
    # Línea teórica y = x²
    x_teorico = np.linspace(x_test.min(), x_test.max(), 100)
    y_teorico = x_teorico ** 2
    plt.plot(x_teorico, y_teorico, 'g--', linewidth=2, label='y = x² (teórico)', alpha=0.7)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Gráfica 2: Residuos (errores)
    plt.subplot(1, 2, 2)
    residuos = y_test_sorted - y_pred_sorted
    plt.scatter(x_sorted, residuos, alpha=0.5, s=20, color='purple')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Residuo (y_real - y_pred)', fontsize=12)
    plt.title('Análisis de Residuos', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Añadir estadísticas de error
    mse = np.mean(residuos ** 2)
    mae = np.mean(np.abs(residuos))
    plt.text(0.05, 0.95, f'MSE: {mse:.6f}\nMAE: {mae:.6f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica de predicciones guardada: {ruta_guardado}")
    plt.close()


def graficar_curvas_aprendizaje(
    history: tf.keras.callbacks.History,
    ruta_guardado: str = "loss_vs_epochs.png"
) -> None:
    """
    Genera y guarda gráficas de las curvas de aprendizaje.
    """
    plt.figure(figsize=(14, 5))
    
    # Gráfica 1: Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Entrenamiento', linewidth=2, color='blue')
    plt.plot(history.history['val_loss'], label='Validación', linewidth=2, color='orange')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida (MSE)', fontsize=12)
    plt.title('Curva de Aprendizaje - Pérdida', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Marcar el mínimo de validación
    min_val_loss = min(history.history['val_loss'])
    min_epoch = history.history['val_loss'].index(min_val_loss)
    plt.plot(min_epoch, min_val_loss, 'r*', markersize=15, 
             label=f'Mejor modelo (época {min_epoch+1})')
    plt.legend(loc='best')
    
    # Gráfica 2: MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Entrenamiento', linewidth=2, color='blue')
    plt.plot(history.history['val_mae'], label='Validación', linewidth=2, color='orange')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Curva de Aprendizaje - MAE', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica de curvas de aprendizaje guardada: {ruta_guardado}")
    plt.close()


def evaluar_modelo(
    modelo: ModeloCuadratico,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evalúa el rendimiento del modelo en el conjunto de prueba.
    """
    y_pred = modelo.predecir(x_test)
    
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metricas = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    return metricas


def imprimir_metricas(metricas: dict) -> None:
    """
    Imprime las métricas de evaluación de forma formateada.
    """
    print(f"\n{'='*60}")
    print(f"MÉTRICAS DE EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print(f"{'='*60}")
    print(f"  MSE (Mean Squared Error):     {metricas['mse']:.8f}")
    print(f"  MAE (Mean Absolute Error):    {metricas['mae']:.8f}")
    print(f"  RMSE (Root Mean Squared Error): {metricas['rmse']:.8f}")
    print(f"  R² (Coeficiente de determinación): {metricas['r2']:.8f}")
    print(f"{'='*60}\n")


def main():
    """
    Función principal que ejecuta el flujo completo de entrenamiento.
    """
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE RED NEURONAL PARA APROXIMACIÓN DE y = x²")
    print("="*60 + "\n")
    
    configurar_semillas(seed=42)
    
    dir_resultados = "."
    
    print("Paso 1: Inicializando modelo...")
    modelo = ModeloCuadratico()
    print("✓ Modelo inicializado\n")
    
    print("Paso 2: Generando datos de entrenamiento...")
    x_total, y_total = modelo.generar_datos(
        n_samples=1000,
        rango=(-1, 1),
        ruido=0.02,
        seed=42
    )
    print()
    
    print("Paso 3: Dividiendo datos en entrenamiento y prueba...")
    x_train, x_test, y_train, y_test = train_test_split(
        x_total, y_total,
        test_size=0.2,
        random_state=42
    )
    
    modelo.x_train = x_train
    modelo.y_train = y_train
    
    print(f"✓ División completada:")
    print(f"  - Entrenamiento: {len(x_train)} muestras ({len(x_train)/len(x_total)*100:.0f}%)")
    print(f"  - Prueba: {len(x_test)} muestras ({len(x_test)/len(x_total)*100:.0f}%)")
    print()
    
    print("Paso 4: Construyendo arquitectura de la red neuronal...")
    modelo.construir_modelo()
    print()
    
    print("Paso 5: Entrenando el modelo...")
    history = modelo.entrenar(
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    print("Paso 6: Evaluando modelo en conjunto de prueba...")
    metricas = evaluar_modelo(modelo, x_test, y_test)
    imprimir_metricas(metricas)
    
    print("Paso 7: Generando predicciones...")
    y_pred_test = modelo.predecir(x_test)
    print(f"✓ Predicciones generadas para {len(x_test)} muestras\n")
    
    print("Paso 8: Generando visualizaciones...")
    
    ruta_pred = os.path.join(dir_resultados, "prediccion_vs_real.png")
    graficar_predicciones(x_test, y_test, y_pred_test, ruta_pred)
    
    ruta_loss = os.path.join(dir_resultados, "loss_vs_epochs.png")
    graficar_curvas_aprendizaje(history, ruta_loss)
    print()
    
    print("Paso 9: Guardando modelo entrenado...")
    path_keras = os.path.join(dir_resultados, "modelo_entrenado.keras")
    path_pkl = os.path.join(dir_resultados, "modelo_entrenado.pkl")
    modelo.guardar_modelo(path_keras, path_pkl)
    print()
    
    print("Paso 10: Verificando carga del modelo...")
    modelo_cargado = ModeloCuadratico()
    modelo_cargado.cargar_modelo(path_tf=path_keras)
    
    y_pred_cargado = modelo_cargado.predecir(x_test[:5])
    y_pred_original = modelo.predecir(x_test[:5])
    
    if np.allclose(y_pred_cargado, y_pred_original):
        print("✓ Verificación exitosa: el modelo cargado produce predicciones idénticas\n")
    else:
        print("⚠ Advertencia: las predicciones difieren después de cargar el modelo\n")
    
    print("Paso 11: Ejemplos de predicciones...")
    print(f"\n{'='*60}")
    print(f"EJEMPLOS DE PREDICCIONES")
    print(f"{'='*60}")
    print(f"{'x':>10} {'y_real':>15} {'y_pred':>15} {'error':>15}")
    print(f"{'-'*60}")
    
    ejemplos_x = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    ejemplos_pred = modelo.predecir(ejemplos_x)
    
    for x_val, y_pred in zip(ejemplos_x, ejemplos_pred):
        y_real = x_val[0] ** 2
        error = abs(y_real - y_pred[0])
        print(f"{x_val[0]:>10.2f} {y_real:>15.6f} {y_pred[0]:>15.6f} {error:>15.6f}")
    
    print(f"{'='*60}\n")
    
    print("Paso 12: Generando resumen del modelo...")
    modelo.resumen()
    
    print(f"\n{'='*60}")
    print(f"✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"\nArchivos generados:")
    print(f"  1. {path_keras}")
    print(f"  2. {path_pkl}")
    print(f"  3. {ruta_pred}")
    print(f"  4. {ruta_loss}")
    print(f"\nPara usar el modelo entrenado:")
    print(f"  >>> from modelo_cuadratico import ModeloCuadratico")
    print(f"  >>> modelo = ModeloCuadratico()")
    print(f"  >>> modelo.cargar_modelo(path_tf='{path_keras}')")
    print(f"  >>> prediccion = modelo.predecir(np.array([[0.5]]))")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Entrenamiento interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
