#!/usr/bin/env python3
"""
Script para ejecutar todos los proyectos del guion
Ejecuta los 12 proyectos principales explicados en el video
"""

import subprocess
import sys
import os

PROYECTOS = [
    ('p0_regresion_cuadratica.py', 'P0: Regresion Cuadratica'),
    ('p1_regresion_multilineal.py', 'P1: Regresion Multilineal'),
    ('p2_clasificacion_fraude.py', 'P2: Clasificacion Fraude'),
    ('p3_multiclase_diagnostico.py', 'P3: Multiclase Diagnostico'),
    ('p4_clustering_kmeans.py', 'P4: Clustering K-Means'),
    ('p5_compresor_imagenes.py', 'P5: Compresor Imagenes'),
    ('p6_cnn_digitos.py', 'P6: CNN Digitos'),
    ('p7_audio_conv1d.py', 'P7: Audio Conv1D'),
    ('p8_detector_objetos.py', 'P8: Detector Objetos'),
    ('p9_segmentador_unet.py', 'P9: Segmentador U-Net'),
    ('p10_lstm_series.py', 'P10: LSTM Series Temporales'),
    ('p11_nlp_sentimientos.py', 'P11: NLP Sentimientos'),
    ('p12_vae_generador.py', 'P12: VAE Generador'),
]


def ejecutar_todos():
    print("\n" + "="*70)
    print(" "*15 + "EJECUTAR TODOS LOS PROYECTOS DEL GUION")
    print("="*70)
    
    for archivo, nombre in PROYECTOS:
        ruta = os.path.join(os.path.dirname(__file__), archivo)
        if os.path.exists(ruta):
            print(f"\n[*] Ejecutando {nombre}...")
            try:
                subprocess.run([sys.executable, ruta], check=True, timeout=60)
            except subprocess.TimeoutExpired:
                print(f"[!] Timeout ejecutando {nombre}")
            except Exception as e:
                print(f"[!] Error ejecutando {nombre}: {e}")
        else:
            print(f"[!] Archivo no encontrado: {ruta}")
    
    print("\n" + "="*70)
    print("Todos los proyectos completados!")
    print("="*70 + "\n")


if __name__ == '__main__':
    ejecutar_todos()
