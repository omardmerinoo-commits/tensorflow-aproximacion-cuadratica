"""Script ejecución Proyecto 8 - Clasificación de Música."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

from clasificador_musica import ExtractorCaracteristicasAudio, ClasificadorMusica


def main():
    print("=" * 70)
    print("CLASIFICADOR DE GENEROS MUSICALES")
    print("=" * 70)
    
    # Generar datos
    print("\n1. Generando datos sintéticos...")
    generos = ['rock', 'clasica', 'pop']
    caracteristicas_lista = []
    etiquetas = []
    
    extractor = ExtractorCaracteristicasAudio()
    
    for genero_idx, genero in enumerate(generos):
        print(f"   Generando {genero}...")
        for _ in range(100):
            y = extractor.generar_audio_sintetico(duracion=1.0, genero=genero)
            caract = extractor.extraer_caracteristicas(y)
            caracteristicas_lista.append(caract)
            etiquetas.append(genero_idx)
    
    X = np.array(caracteristicas_lista, dtype=np.float32)
    y = np.array(etiquetas, dtype=np.int32)
    print(f"   Total: {len(X)} muestras, {X.shape[1]} características")
    
    # Normalizar
    print("\n2. Normalizando...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    print("\n3. Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Entrenar
    print("\n4. Entrenando modelo...")
    modelo = ClasificadorMusica(input_dim=20, num_generos=3)
    modelo.construir_modelo()
    history = modelo.entrenar(X_train, y_train, X_val=X_test, y_val=y_test,
                             epochs=50, verbose=0)
    print(f"   Épocas: {history['epochs']}, Accuracy: {history['accuracy_final']:.4f}")
    
    # Evaluar
    print("\n5. Evaluando...")
    metricas = modelo.evaluar(X_test, y_test)
    print(f"   Accuracy test: {metricas['accuracy']:.4f}")
    
    # Guardar
    print("\n6. Guardando...")
    modelo.guardar_modelo()
    
    # Reporte
    reporte = {
        'proyecto': 'Clasificador de Géneros Musicales',
        'generos': ['Rock', 'Clásica', 'Pop'],
        'dataset': {'muestras': 300, 'caracteristicas': 20},
        'metricas': metricas
    }
    with open('REPORTE_MUSICA.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    main()
