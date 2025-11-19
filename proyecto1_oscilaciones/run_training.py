"""
Script de entrenamiento automático para Oscilaciones Amortiguadas.
Ejecuta el flujo completo de generación, entrenamiento y evaluación.
"""

from oscilaciones_amortiguadas import OscilacionesAmortiguadas
import numpy as np
import sys


def main():
    """Ejecuta el entrenamiento completo."""
    
    print("\n" + "="*80)
    print("🌊 ENTRENAMIENTO DE MODELO: OSCILACIONES AMORTIGUADAS")
    print("="*80 + "\n")
    
    try:
        # 1. Crear modelo
        print("📦 Creando instancia del modelo...")
        modelo = OscilacionesAmortiguadas(seed=42)
        print("✅ Modelo creado\n")
        
        # 2. Generar datos
        print("📊 Generando datos sintéticos...")
        X_train, X_test, y_train, y_test = modelo.generar_datos(
            num_muestras=1000,
            tiempo_max=10.0,
            puntos_tiempo=100,
            ruido=0.02,
            test_size=0.2
        )
        print(f"✅ Datos generados:")
        print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   - Prueba: {X_test.shape[0]} muestras\n")
        
        # 3. Construir modelo
        print("🏗️  Construyendo arquitectura de red neuronal...")
        modelo.construir_modelo(
            input_shape=7,
            capas_ocultas=[256, 128, 64, 32],
            tasa_aprendizaje=0.001,
            dropout_rate=0.2
        )
        print(f"✅ Red construida con {modelo.config['parametros_totales']} parámetros\n")
        
        # 4. Entrenar
        print("🎯 Entrenando modelo (esto puede tomar un momento)...")
        info = modelo.entrenar(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            early_stopping_patience=10,
            verbose=1
        )
        print(f"\n✅ Entrenamiento completado:")
        print(f"   - Épocas: {info['epochs_entrenadas']}")
        print(f"   - Loss final: {info['loss_final']:.6f}\n")
        
        # 5. Evaluar
        print("📈 Evaluando modelo...")
        metricas = modelo.evaluar()
        print(f"✅ Métricas de Prueba:")
        print(f"   - MSE:  {metricas['mse']:.6f}")
        print(f"   - RMSE: {metricas['rmse']:.6f}")
        print(f"   - MAE:  {metricas['mae']:.6f}")
        print(f"   - R²:   {metricas['r2']:.4f}\n")
        
        # 6. Validación cruzada
        print("🔄 Realizando validación cruzada (5-fold)...")
        cv_results = modelo.validacion_cruzada(
            X_train, y_train,
            k_folds=5,
            epochs=50
        )
        print(f"✅ Validación Cruzada:")
        print(f"   - R² promedio:  {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
        print(f"   - MAE promedio: {cv_results['mae_mean']:.6f} ± {cv_results['mae_std']:.6f}\n")
        
        # 7. Visualizar
        print("🎨 Creando visualizaciones...")
        modelo.visualizar_predicciones(salida='oscilaciones_predicciones.png')
        print()
        
        # 8. Guardar
        print("💾 Guardando modelo...")
        modelo.guardar_modelo('oscilaciones_modelo_entrenado')
        print()
        
        # 9. Resumen final
        print("="*80)
        print("📋 RESUMEN FINAL")
        print("="*80)
        resumen = modelo.resumen_modelo()
        for key, value in resumen.items():
            if key != 'configuración':
                print(f"  {key}: {value}")
        print()
        
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE!\n")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
