"""Script ejecución Proyecto 7 - Predicción de Materiales."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt

from predictor_materiales import GeneradorDatosMateriales, PredictorMateriales


def main():
    print("=" * 70)
    print("PREDICTOR DE PROPIEDADES DE MATERIALES")
    print("=" * 70)
    
    # Datos
    print("\n1. Generando datos...")
    gen = GeneradorDatosMateriales(seed=42)
    X, y = gen.generar_composiciones(n_muestras=500)
    print(f"   Muestras: {X.shape[0]}, Características: {X.shape[1]}")
    
    # Normalizar
    print("\n2. Normalizando...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split
    print("\n3. Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Entrenar
    print("\n4. Entrenando modelo...")
    modelo = PredictorMateriales(input_dim=5, output_dim=3)
    modelo.construir_modelo()
    history = modelo.entrenar(X_train, y_train, X_val=X_test, y_val=y_test, 
                              epochs=100, verbose=0)
    print(f"   Épocas: {history['epochs']}, Loss: {history['loss_final']:.6f}")
    
    # Evaluar
    print("\n5. Evaluando...")
    metricas = modelo.evaluar(X_test, y_test)
    print(f"   Loss: {metricas['loss']:.6f}, MAE: {metricas['mae']:.6f}")
    
    # Guardar
    print("\n6. Guardando modelo...")
    modelo.guardar_modelo()
    
    # Reporte
    print("\n7. Generando reporte...")
    reporte = {
        'proyecto': 'Predictor de Materiales',
        'dataset': {'muestras': 500, 'caracteristicas': 5, 'propiedades': 3},
        'metricas': metricas,
        'historia': history
    }
    with open('REPORTE_MATERIALES.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    main()
