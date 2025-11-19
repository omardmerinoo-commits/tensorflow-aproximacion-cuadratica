"""
🚀 Script de Entrenamiento: Simulador de Qubits
================================================

Ejemplo completo de uso del simulador:
1. Crear simulador
2. Generar datos de evolución cuántica
3. Entrenar red neuronal
4. Evaluar modelo
5. Hacer predicciones
6. Guardar modelo

Ejecución:
    python run_training.py
"""

import numpy as np
import matplotlib.pyplot as plt
from simulador_qubit import SimuladorQubit
from pathlib import Path

def main():
    """Flujo completo de entrenamiento."""
    
    print("\n" + "="*80)
    print("🎯 SIMULADOR DE QUBITS - ENTRENAMIENTO COMPLETO")
    print("="*80)
    
    # ========================================================================
    # PASO 1: Crear Simulador
    # ========================================================================
    
    print("\n📊 PASO 1: Crear Simulador")
    print("-" * 80)
    
    sim = SimuladorQubit(num_qubits=1, seed=42)
    print(f"✅ Simulador creado: {sim.num_qubits} qubit(s)")
    print(f"   Dimensión del espacio: {sim.dim}")
    print(f"   Estado inicial: |0⟩")
    
    # ========================================================================
    # PASO 2: Verificar Puertas Cuánticas
    # ========================================================================
    
    print("\n🔀 PASO 2: Demostración de Puertas")
    print("-" * 80)
    
    # Crear superposición
    sim.puerta_hadamard()
    probs = sim.get_probabilidades()
    print(f"✅ Hadamard aplicada → Superposición")
    print(f"   P(0) = {probs[0]:.4f}")
    print(f"   P(1) = {probs[1]:.4f}")
    
    # Medir múltiples veces
    sim.estado.amplitudes = (sim.estado.amplitudes * 0 + 1/np.sqrt(2) + 1j*0).reshape(-1, 1)
    resultados = [sim.medir(seed=i) for i in range(100)]
    print(f"   Mediciones (100x): {resultados.count(0)} ceros, {resultados.count(1)} unos")
    
    # ========================================================================
    # PASO 3: Generar Datos
    # ========================================================================
    
    print("\n📈 PASO 3: Generar Datos de Evolución")
    print("-" * 80)
    
    X_train, y_train, X_test, y_test = sim.generar_datos_evolucion(
        num_muestras=500,
        pasos_tiempo=10,
        test_size=0.2,
        seed=42
    )
    
    print(f"✅ Datos generados")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_test: {y_test.shape}")
    
    # ========================================================================
    # PASO 4: Construir Modelo
    # ========================================================================
    
    print("\n🔨 PASO 4: Construir Modelo Neural")
    print("-" * 80)
    
    modelo = sim.construir_modelo(
        capas_ocultas=[128, 64, 32],
        tasa_aprendizaje=0.001,
        dropout_rate=0.2
    )
    
    print(f"✅ Modelo construido")
    print(f"   Parámetros: {modelo.count_params():,}")
    print(f"   Capas: {len(modelo.layers)}")
    modelo.summary()
    
    # ========================================================================
    # PASO 5: Entrenar Modelo
    # ========================================================================
    
    print("\n🚀 PASO 5: Entrenar Modelo")
    print("-" * 80)
    
    historial = sim.entrenar(
        X_train, y_train,
        X_test, y_test,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    print(f"✅ Entrenamiento completado")
    print(f"   Épocas: {len(historial['loss'])}")
    print(f"   Pérdida inicial: {historial['loss'][0]:.6f}")
    print(f"   Pérdida final: {historial['loss'][-1]:.6f}")
    
    # ========================================================================
    # PASO 6: Evaluar Modelo
    # ========================================================================
    
    print("\n📊 PASO 6: Evaluar Modelo")
    print("-" * 80)
    
    metricas = sim.evaluar(X_test, y_test)
    
    print(f"✅ Evaluación completada")
    print(f"   MSE: {metricas['mse']:.6f}")
    print(f"   RMSE: {metricas['rmse']:.6f}")
    print(f"   MAE: {metricas['mae']:.6f}")
    print(f"   Fidelidad promedio: {metricas['fidelidad_promedio']:.6f}")
    print(f"   Muestras: {metricas['samples']}")
    
    # ========================================================================
    # PASO 7: Hacer Predicciones
    # ========================================================================
    
    print("\n🔮 PASO 7: Predicciones de Evolución")
    print("-" * 80)
    
    estado_inicial = np.array([1.0, 0.0], dtype=np.float32)
    predicciones = sim.predecir_evolucion(estado_inicial, pasos=5)
    
    print(f"✅ Predicciones realizadas")
    print(f"   Pasos predichos: {len(predicciones)}")
    for i, pred in enumerate(predicciones):
        print(f"   Paso {i+1}: amplitudes = [{pred[0].real:.4f}, {pred[1].real:.4f}]")
    
    # ========================================================================
    # PASO 8: Guardar Modelo
    # ========================================================================
    
    print("\n💾 PASO 8: Guardar Modelo")
    print("-" * 80)
    
    ruta_modelo = Path("modelos") / "simulador_completo"
    resultado = sim.guardar_modelo(str(ruta_modelo))
    
    if resultado:
        print(f"✅ Modelo guardado")
        print(f"   Ubicación: {ruta_modelo}")
    else:
        print(f"❌ Error guardando modelo")
    
    # ========================================================================
    # PASO 9: Visualizar Resultados
    # ========================================================================
    
    print("\n📈 PASO 9: Visualizar Resultados")
    print("-" * 80)
    
    plt.figure(figsize=(14, 5))
    
    # Gráfico 1: Pérdida de entrenamiento
    plt.subplot(1, 3, 1)
    plt.plot(historial['loss'], label='Pérdida entrenamiento', linewidth=2)
    if 'val_loss' in historial:
        plt.plot(historial['val_loss'], label='Pérdida validación', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida vs Época')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Métricas
    plt.subplot(1, 3, 2)
    metricas_nombres = ['MSE', 'RMSE', 'MAE']
    metricas_valores = [
        metricas['mse'],
        metricas['rmse'],
        metricas['mae']
    ]
    plt.bar(metricas_nombres, metricas_valores, color=['red', 'green', 'blue'], alpha=0.7)
    plt.ylabel('Error')
    plt.title('Métricas de Evaluación')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 3: Fidelidad
    plt.subplot(1, 3, 3)
    fidelidad = metricas['fidelidad_promedio']
    plt.text(0.5, 0.5, f"Fidelidad\n{fidelidad:.4f}", 
             ha='center', va='center', fontsize=20, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Fidelidad Cuántica Promedio')
    
    plt.tight_layout()
    plt.savefig('resultados_entrenamiento.png', dpi=150)
    print(f"✅ Gráfico guardado: resultados_entrenamiento.png")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO CON ÉXITO")
    print("="*80)
    print("\n📊 RESUMEN:")
    print(f"   ✓ Simulador creado con {sim.num_qubits} qubit(s)")
    print(f"   ✓ {len(X_train)} muestras de entrenamiento generadas")
    print(f"   ✓ Modelo con {modelo.count_params():,} parámetros")
    print(f"   ✓ Entrenado en {len(historial['loss'])} épocas")
    print(f"   ✓ Fidelidad final: {metricas['fidelidad_promedio']:.6f}")
    print(f"   ✓ Modelo guardado en: {ruta_modelo}")
    print(f"   ✓ Gráficos guardados")
    print("\n🎓 Próximos pasos:")
    print("   - Ejecutar pruebas: pytest test_simulador_qubit.py -v")
    print("   - Ver documentación: cat README.md")
    print("   - Modificar parámetros y experimentar")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
