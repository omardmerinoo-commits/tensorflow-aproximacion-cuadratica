"""
Aplicación: Predictor de Precios de Casas
==========================================

Caso de uso real: Predicción de precios de inmuebles usando regresión cuadrática.

Características:
- Entrenamiento con datos de mercado inmobiliario
- Predicción de precios basada en superficie y características
- Validación de modelos
- Generación de reportes

Autor: Proyecto TensorFlow
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from datetime import datetime
from pathlib import Path


class GeneradorDatosCasas:
    """Generador de datos sintéticos de mercado inmobiliario."""
    
    def __init__(self, seed=42):
        """Inicializa el generador."""
        np.random.seed(seed)
        self.seed = seed
    
    def generar_dataset(self, n_samples=200):
        """
        Genera dataset sintético de precios de casas.
        
        Args:
            n_samples: Número de muestras
        
        Returns:
            Dict con X (superficie), y (precio)
        """
        # Superficie en m² (50-500)
        X = np.random.uniform(50, 500, n_samples)
        
        # Relación cuadrática: precio = base + coef1*X + coef2*X²
        # Base de precio: $50,000
        # Coeficientes realistas
        base = 50000
        coef1 = 1000  # Aumento lineal
        coef2 = 5     # Aumento cuadrático (economías de escala)
        noise = np.random.normal(0, 10000, n_samples)
        
        y = base + coef1 * X + coef2 * (X ** 2) + noise
        
        # Precios realistas (en dólares)
        y = np.maximum(y, 30000)  # Precio mínimo
        
        return {
            'X': X.reshape(-1, 1),
            'y': y,
            'features': ['Superficie (m²)']
        }
    
    def agregar_caracteristicas(self, X, n_features=3):
        """
        Agrega características adicionales.
        
        Args:
            X: Datos originales (n_samples, 1)
            n_features: Número de características a agregar
        
        Returns:
            Array con características adicionales
        """
        n_samples = X.shape[0]
        X_extended = X.copy()
        
        # Antigüedad (años)
        antiguedad = np.random.uniform(0, 50, n_samples).reshape(-1, 1)
        X_extended = np.hstack([X_extended, antiguedad])
        
        # Número de habitaciones
        habitaciones = np.random.randint(1, 6, n_samples).reshape(-1, 1)
        X_extended = np.hstack([X_extended, habitaciones])
        
        return X_extended


class PredictorPreciosCasas:
    """Modelo de predicción de precios de casas."""
    
    def __init__(self, seed=42):
        """Inicializa el predictor."""
        np.random.seed(seed)
        self.seed = seed
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.modelo = None
        self.coeficientes = None
        self.metricas = {}
    
    def fit(self, X, y):
        """
        Entrena modelo de regresión cuadrática.
        
        Args:
            X: Features (n_samples, 1)
            y: Targets (n_samples,)
        """
        # Preprocesamiento
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Crear características polinomiales
        X_poly = np.column_stack([
            np.ones(len(X_scaled)),
            X_scaled,
            X_scaled ** 2
        ])
        
        # Regresión lineal: solve (X^T X) β = X^T y
        self.coeficientes = np.linalg.lstsq(X_poly, y_scaled, rcond=None)[0]
        
        print(f"✅ Modelo entrenado")
        print(f"   Coeficientes: {self.coeficientes}")
    
    def predict(self, X):
        """
        Realiza predicciones.
        
        Args:
            X: Features (n_samples, 1)
        
        Returns:
            Predicciones en escala original
        """
        X_scaled = self.scaler_X.transform(X)
        
        X_poly = np.column_stack([
            np.ones(len(X_scaled)),
            X_scaled,
            X_scaled ** 2
        ])
        
        y_scaled = X_poly @ self.coeficientes
        y = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        
        return y
    
    def evaluar(self, X, y):
        """
        Evalúa el modelo.
        
        Args:
            X: Features
            y: Targets verdaderos
        """
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        self.metricas = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        print(f"\n📊 Métricas de evaluación:")
        print(f"   MSE:  ${mse:,.0f}")
        print(f"   RMSE: ${rmse:,.0f}")
        print(f"   MAE:  ${mae:,.0f}")
        print(f"   R²:   {r2:.4f}")
        
        return self.metricas
    
    def predecir_caso(self, superficie):
        """
        Predice precio para una superficie específica.
        
        Args:
            superficie: Superficie en m²
        
        Returns:
            Precio predicho
        """
        X_test = np.array([[superficie]])
        precio = self.predict(X_test)[0]
        return precio


def main():
    """Demostración de la aplicación."""
    print("\n" + "="*80)
    print("🏠 PREDICTOR DE PRECIOS DE CASAS - REGRESIÓN CUADRÁTICA")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando dataset de mercado inmobiliario...")
    generador = GeneradorDatosCasas(seed=42)
    datos = generador.generar_dataset(n_samples=200)
    
    X = datos['X']
    y = datos['y']
    
    print(f"✅ Dataset generado: {len(X)} muestras")
    print(f"   Superficie: [{X.min():.0f}, {X.max():.0f}] m²")
    print(f"   Precio: [${y.min():,.0f}, ${y.max():,.0f}]")
    
    # Paso 2: Split train/test
    print("\n[2] División train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Paso 3: Entrenar modelo
    print("\n[3] Entrenando modelo...")
    modelo = PredictorPreciosCasas()
    modelo.fit(X_train, y_train)
    
    # Paso 4: Evaluar en test
    print("\n[4] Evaluando en conjunto de test...")
    metricas = modelo.evaluar(X_test, y_test)
    
    # Paso 5: Predicciones ejemplo
    print("\n[5] Predicciones de ejemplo:")
    superficies_prueba = [100, 200, 300, 400, 500]
    
    for sup in superficies_prueba:
        precio = modelo.predecir_caso(sup)
        print(f"   {sup:3d} m² → ${precio:,.0f}")
    
    # Paso 6: Análisis de residuos
    print("\n[6] Análisis de residuos...")
    y_pred_train = modelo.predict(X_train)
    residuos = y_train - y_pred_train
    
    print(f"   Mean residual: ${residuos.mean():,.0f}")
    print(f"   Std residual:  ${residuos.std():,.0f}")
    print(f"   Max error:     ${np.abs(residuos).max():,.0f}")
    
    # Paso 7: Generar reporte
    print("\n[7] Generando reporte...")
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'Regresión Cuadrática',
        'muestras': len(X),
        'características': datos['features'],
        'metricas': metricas,
        'predicciones_ejemplo': {
            f"{sup}m2": float(modelo.predecir_caso(sup))
            for sup in superficies_prueba
        }
    }
    
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte_file = output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(reporte_file, 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte guardado: {reporte_file.name}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
