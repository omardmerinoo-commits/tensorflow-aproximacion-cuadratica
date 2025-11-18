"""Script ejecución Proyecto 9 - Visión Computacional."""

import numpy as np
from sklearn.model_selection import train_test_split
import json

from contador_objetos import GeneradorImagenesSinteticas, ContadorObjetos


def main():
    print("=" * 70)
    print("CONTADOR DE OBJETOS - VISION COMPUTACIONAL")
    print("=" * 70)
    
    # Generar imágenes
    print("\n1. Generando imágenes sintéticas...")
    imagenes, conteos = GeneradorImagenesSinteticas.generar_dataset(
        num_muestras=300, tamanio=64
    )
    print(f"   Imágenes: {imagenes.shape}")
    print(f"   Conteos: min={conteos.min()}, max={conteos.max()}, mean={conteos.mean():.2f}")
    
    # Split
    print("\n2. Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        imagenes, conteos, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Entrenar
    print("\n3. Entrenando CNN...")
    modelo = ContadorObjetos(tamanio_imagen=64)
    modelo.construir_modelo()
    history = modelo.entrenar(X_train, y_train, X_val=X_test, y_val=y_test,
                             epochs=100, verbose=0)
    print(f"   Épocas: {history['epochs']}, Loss: {history['loss_final']:.6f}")
    
    # Evaluar
    print("\n4. Evaluando...")
    metricas = modelo.evaluar(X_test, y_test)
    print(f"   Loss: {metricas['loss']:.6f}, MAE: {metricas['mae']:.6f}")
    
    # Predicciones
    print("\n5. Realizando predicciones...")
    predicciones = modelo.predecir(X_test[:20])
    print(f"   Primeras 20 predicciones: {predicciones}")
    print(f"   Valores reales: {y_test[:20].astype(int)}")
    
    # Guardar
    print("\n6. Guardando...")
    modelo.guardar_modelo()
    
    # Reporte
    reporte = {
        'proyecto': 'Contador de Objetos',
        'dataset': {'muestras': 300, 'tamanio_imagen': 64},
        'metricas': metricas
    }
    with open('REPORTE_VISION.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    main()
