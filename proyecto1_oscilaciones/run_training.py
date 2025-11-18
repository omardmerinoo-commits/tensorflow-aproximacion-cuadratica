"""
Script de entrenamiento principal para el modelo de oscilaciones amortiguadas.

Ejecuta el pipeline completo:
1. Generación de datos sintéticos
2. División train/test
3. Entrenamiento del modelo
4. Validación cruzada k-fold
5. Análisis de resultados
6. Serialización de modelos
"""

import numpy as np
import matplotlib.pyplot as plt
from oscilaciones_amortiguadas import OscilacionesAmortiguadas
from pathlib import Path
import json
import time

# Configurar backend de matplotlib para entorno sin GUI
plt.switch_backend('Agg')


def main():
    """Ejecuta el pipeline completo de entrenamiento."""
    
    print("=" * 70)
    print("PROYECTO 1: Red neuronal para Oscilaciones Amortiguadas")
    print("=" * 70)
    
    # ==================== GENERACIÓN DE DATOS ====================
    print("\n[1/5] Generando datos sintéticos...")
    inicio = time.time()
    
    modelo_oia = OscilacionesAmortiguadas(seed=42)
    X, y = modelo_oia.generar_datos(
        num_muestras=500,
        tiempo_max=10.0,
        ruido_sigma=0.02
    )
    
    tiempo_gen = time.time() - inicio
    print(f"✓ Datos generados: X{X.shape}, y{y.shape}")
    print(f"  Tiempo: {tiempo_gen:.2f}s")
    
    # Dividir en train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # ==================== ENTRENAMIENTO ====================
    print("\n[2/5] Entrenando el modelo...")
    inicio = time.time()
    
    info_entrenamiento = modelo_oia.entrenar(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        early_stopping_patience=15,
        verbose=1
    )
    
    tiempo_entrenamiento = time.time() - inicio
    print(f"✓ Entrenamiento completado")
    print(f"  Épocas entrenadas: {info_entrenamiento['epochs_entrenadas']}")
    print(f"  Loss final: {info_entrenamiento['loss_final']:.6f}")
    print(f"  Val Loss final: {info_entrenamiento['val_loss_final']:.6f}")
    print(f"  MAE final: {info_entrenamiento['mae_final']:.6f}")
    print(f"  Tiempo: {tiempo_entrenamiento:.2f}s")
    
    # ==================== VALIDACIÓN CRUZADA ====================
    print("\n[3/5] Realizando validación cruzada (5-fold)...")
    inicio = time.time()
    
    cv_results = modelo_oia.validacion_cruzada(
        X_train, y_train,
        k_folds=5,
        epochs=50
    )
    
    tiempo_cv = time.time() - inicio
    print(f"✓ Validación cruzada completada")
    print(f"  MSE: {cv_results['mse_mean']:.6f} ± {cv_results['mse_std']:.6f}")
    print(f"  MAE: {cv_results['mae_mean']:.6f} ± {cv_results['mae_std']:.6f}")
    print(f"  R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    print(f"  Tiempo: {tiempo_cv:.2f}s")
    
    # ==================== EVALUACIÓN EN TEST ====================
    print("\n[4/5] Evaluando en conjunto de prueba...")
    
    y_pred = modelo_oia.predecir(X_test)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse_test = mean_squared_error(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    
    print(f"✓ Métricas en test set:")
    print(f"  MSE: {mse_test:.6f}")
    print(f"  MAE: {mae_test:.6f}")
    print(f"  R²: {r2_test:.4f}")
    
    # ==================== GUARDADO DE MODELOS ====================
    print("\n[5/5] Guardando modelos y resultados...")
    
    # Crear directorio de salida
    output_dir = Path('resultados_entrenamiento')
    output_dir.mkdir(exist_ok=True)
    
    # Guardar modelo
    modelo_oia.guardar_modelo(str(output_dir / 'modelo_oscilaciones.keras'))
    
    # Guardar resultados
    resultados = {
        'configuracion': {
            'num_muestras_train': len(X_train),
            'num_muestras_test': len(X_test),
            'features': ['tiempo', 'masa', 'amortiguamiento', 'rigidez', 'pos_inicial', 'vel_inicial', 'zeta']
        },
        'entrenamiento': info_entrenamiento,
        'validacion_cruzada': {
            'mse_mean': cv_results['mse_mean'],
            'mae_mean': cv_results['mae_mean'],
            'r2_mean': cv_results['r2_mean']
        },
        'test': {
            'mse': float(mse_test),
            'mae': float(mae_test),
            'r2': float(r2_test)
        },
        'tiempos_segundos': {
            'generacion_datos': tiempo_gen,
            'entrenamiento': tiempo_entrenamiento,
            'validacion_cruzada': tiempo_cv
        }
    }
    
    with open(output_dir / 'resultados.json', 'w') as f:
        json.dump(resultados, f, indent=4)
    
    # ==================== VISUALIZACIÓN ====================
    print("\n[EXTRA] Generando visualizaciones...")
    
    # Gráfica 1: Historia de entrenamiento
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(modelo_oia.history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(modelo_oia.history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Evolución del Loss durante el Entrenamiento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(modelo_oia.history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(modelo_oia.history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Evolución del MAE durante el Entrenamiento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'historia_entrenamiento.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: historia_entrenamiento.png")
    
    # Gráfica 2: Predicciones vs valores reales
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Valores Reales')
    axes[0].set_ylabel('Predicciones')
    axes[0].set_title(f'Predicciones vs Reales (R² = {r2_test:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # Residuos
    residuos = y_test.flatten() - y_pred.flatten()
    axes[1].scatter(y_pred, residuos, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicciones')
    axes[1].set_ylabel('Residuos')
    axes[1].set_title('Análisis de Residuos')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predicciones_analisis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: predicciones_analisis.png")
    
    # Gráfica 3: Validación cruzada
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    folds = range(1, len(cv_results['scores_por_fold']['mse']) + 1)
    
    axes[0].bar(folds, cv_results['scores_por_fold']['mse'])
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE por Fold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(folds, cv_results['scores_por_fold']['mae'])
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE por Fold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(folds, cv_results['scores_por_fold']['r2'])
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² por Fold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validacion_cruzada.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: validacion_cruzada.png")
    
    # ==================== RESUMEN FINAL ====================
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"Resultados guardados en: {output_dir.absolute()}")
    print(f"  - modelo_oscilaciones.keras (Modelo entrenado)")
    print(f"  - modelo_oscilaciones.json (Configuración)")
    print(f"  - resultados.json (Métricas y evaluación)")
    print(f"  - historia_entrenamiento.png")
    print(f"  - predicciones_analisis.png")
    print(f"  - validacion_cruzada.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
