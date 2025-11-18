"""
Script de ejecución para aproximador de funciones no lineales.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

from aproximador_funciones import (
    GeneradorFuncionesNoLineales,
    AproximadorFuncionesNoLineales
)


def main():
    """Ejecutar pipeline de aproximación de funciones."""
    
    print("=" * 70)
    print("APROXIMADOR DE FUNCIONES NO LINEALES")
    print("=" * 70)
    
    funciones = {
        'exponencial_amortiguada': GeneradorFuncionesNoLineales.funcion_exponencial_amortiguada,
        'polinomica_compleja': GeneradorFuncionesNoLineales.funcion_polinomica_compleja,
        'trigonometrica': GeneradorFuncionesNoLineales.funcion_trigonometrica,
        'logaritmica': GeneradorFuncionesNoLineales.funcion_logaritmica,
    }
    
    resultados = {}
    
    for nombre_func, funcion in funciones.items():
        print(f"\n{'=' * 70}")
        print(f"Función: {nombre_func.replace('_', ' ').title()}")
        print('=' * 70)
        
        # Generar datos
        print("\n1. Generando datos...")
        X, y = GeneradorFuncionesNoLineales.generar_datos(
            funcion, x_min=-10.0, x_max=10.0, n_samples=500, ruido=0.02
        )
        print(f"   Muestras: {X.shape[0]}")
        print(f"   Rango X: [{X.min():.2f}, {X.max():.2f}]")
        print(f"   Rango Y: [{y.min():.2f}, {y.max():.2f}]")
        
        # Dividir datos
        print("\n2. Dividiendo datos...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Construir y entrenar
        print("\n3. Construyendo y entrenando modelo...")
        modelo = AproximadorFuncionesNoLineales(seed=42)
        modelo.construir_modelo()
        
        history_dict = modelo.entrenar(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=150,
            batch_size=32,
            verbose=0
        )
        print(f"   Épocas: {history_dict['epochs']}")
        print(f"   Loss final: {history_dict['loss_final']:.6f}")
        print(f"   MAE final: {history_dict['mae_final']:.6f}")
        
        # Evaluar
        print("\n4. Evaluando...")
        metricas = modelo.evaluar(X_test, y_test)
        print(f"   Loss en test: {metricas['loss']:.6f}")
        print(f"   MAE en test: {metricas['mae']:.6f}")
        
        # Predicciones en rango
        print("\n5. Generando predicciones...")
        X_rango = np.linspace(-10, 10, 200).reshape(-1, 1).astype(np.float32)
        y_pred = modelo.predecir(X_rango).flatten()
        y_real = funcion(X_rango.flatten())
        
        error_medio = np.mean(np.abs(y_pred - y_real))
        print(f"   Error medio: {error_medio:.6f}")
        
        # Guardar modelo
        modelo.guardar_modelo(f'modelo_{nombre_func}.keras')
        
        # Gráfico
        _graficar_funcion(
            X, y, X_rango, y_pred, y_real, nombre_func, metricas
        )
        
        resultados[nombre_func] = {
            'metricas': metricas,
            'historia': history_dict,
            'error_medio': float(error_medio)
        }
    
    # Reporte
    print("\n" + "=" * 70)
    print("GUARDANDO REPORTE...")
    print("=" * 70)
    _generar_reporte_completo(resultados)
    print("\nReporte guardado: REPORTE_FUNCIONES.json")
    
    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)


def _graficar_funcion(X, y, X_rango, y_pred, y_real, nombre_func, metricas):
    """Graficar función aproximada."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Puntos de entrenamiento vs aproximación
    axes[0].scatter(X, y, alpha=0.5, s=20, label='Datos (con ruido)')
    axes[0].plot(X_rango, y_pred, 'r-', linewidth=2, label='Red Neuronal')
    axes[0].plot(X_rango, y_real, 'g--', linewidth=2, label='Función real')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'{nombre_func.replace("_", " ").title()}\nMAE: {metricas["mae"]:.6f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error absoluto
    error = np.abs(y_pred - y_real)
    axes[1].plot(X_rango, error, 'purple', linewidth=2)
    axes[1].fill_between(X_rango.flatten(), 0, error, alpha=0.3, color='purple')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Error absoluto')
    axes[1].set_title('Error de aproximación')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'funcion_{nombre_func}.png', dpi=300, bbox_inches='tight')
    plt.close()


def _generar_reporte_completo(resultados):
    """Generar reporte JSON."""
    reporte = {
        'proyecto': 'Aproximador de Funciones No Lineales',
        'num_funciones': len(resultados),
        'funciones': resultados
    }
    
    with open('REPORTE_FUNCIONES.json', 'w') as f:
        json.dump(reporte, f, indent=2)


if __name__ == '__main__':
    main()
