"""
Script de ejecución para entrenamiento y prueba del clasificador de fases.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

from generador_datos_fases import GeneradorDatosFases
from modelo_clasificador_fases import ModeloClasificadorFases


def main():
    """Ejecutar pipeline completo de clasificación de fases."""
    
    print("=" * 60)
    print("CLASIFICADOR DE FASES - PIPELINE DE ENTRENAMIENTO")
    print("=" * 60)
    
    # Generar datos
    print("\n1. Generando datos sintéticos...")
    generador = GeneradorDatosFases(seed=42)
    X, y = generador.generar_datos_completos(n_samples_por_clase=300)
    print(f"   Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"   Distribución de clases: {np.bincount(y)}")
    
    # Normalizar datos
    print("\n2. Normalizando datos...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir datos
    print("\n3. Dividiendo en train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} muestras")
    print(f"   Test: {X_test.shape[0]} muestras")
    
    # Construir modelo
    print("\n4. Construyendo modelo...")
    modelo = ModeloClasificadorFases(input_dim=X.shape[1], num_classes=3)
    modelo.construir_modelo()
    print("   Modelo construido exitosamente")
    
    # Entrenar
    print("\n5. Entrenando modelo...")
    history_dict = modelo.entrenar(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    print(f"   Épocas entrenadas: {history_dict['epochs']}")
    print(f"   Accuracy final: {history_dict['accuracy_final']:.4f}")
    
    # Evaluar
    print("\n6. Evaluando en datos de test...")
    metricas = modelo.evaluar(X_test, y_test)
    print(f"   Loss: {metricas['loss']:.4f}")
    print(f"   Accuracy: {metricas['accuracy']:.4f}")
    
    # Hacer predicciones
    print("\n7. Realizando predicciones...")
    predicciones, probabilidades = modelo.predecir(X_test[:10])
    print("   Primeras 10 predicciones:")
    for i, (pred, probs) in enumerate(zip(predicciones[:10], probabilidades[:10])):
        clase = modelo.etiquetas_clases[pred]
        print(f"   [{i}] {clase}: {probs}")
    
    # Guardar modelo
    print("\n8. Guardando modelo...")
    modelo.guardar_modelo('modelo_fases.keras')
    modelo.guardar_config('config_fases.json')
    print("   Modelo guardado: modelo_fases.keras")
    print("   Configuración guardada: config_fases.json")
    
    # Graficar historia
    print("\n9. Generando gráficos...")
    _generar_graficos(modelo.history)
    print("   Gráficos guardados")
    
    # Guardar reporte
    print("\n10. Generando reporte...")
    _generar_reporte(X.shape, len(y), metricas, history_dict)
    print("   Reporte guardado: REPORTE_FASES.json")
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)


def _generar_graficos(history):
    """Generar gráficos de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Pérdida durante entrenamiento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Precisión durante entrenamiento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entrenamiento_fases.png', dpi=300, bbox_inches='tight')
    plt.close()


def _generar_reporte(shape_X, total_muestras, metricas, history_dict):
    """Generar reporte JSON."""
    reporte = {
        'proyecto': 'Clasificación de Fases',
        'dataset': {
            'total_muestras': total_muestras,
            'caracteristicas': shape_X[1],
            'clases': ['Sólido', 'Líquido', 'Gas'],
            'num_clases': 3
        },
        'metricas': metricas,
        'entrenamiento': history_dict,
        'modelo': {
            'tipo': 'Red Neuronal Profunda (MLP)',
            'capas': 5,
            'parametros': 'Calculados por Keras'
        }
    }
    
    with open('REPORTE_FASES.json', 'w') as f:
        json.dump(reporte, f, indent=2)


if __name__ == '__main__':
    main()
