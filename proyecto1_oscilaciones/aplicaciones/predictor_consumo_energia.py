"""
Aplicación: Análisis de Consumo Energético
===========================================

Caso de uso real: Predicción de consumo eléctrico basado en temperatura y ocupación.

Características:
- Análisis multivariado de consumo
- Predicción de demanda energética
- Optimización de recursos
- Alertas de consumo anómalo

Autor: Proyecto TensorFlow
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class GeneradorDatosEnergia:
    """Generador de datos de consumo energético."""
    
    def __init__(self, seed=42):
        """Inicializa el generador."""
        np.random.seed(seed)
        self.seed = seed
    
    def generar_timeseries(self, dias=30):
        """
        Genera serie temporal de consumo.
        
        Args:
            dias: Número de días de datos
        
        Returns:
            DataFrame con consumo y features
        """
        horas = dias * 24
        fechas = [datetime.now() - timedelta(hours=i) for i in range(horas, 0, -1)]
        
        # Temperatura (°C) - patrón diario
        hora_del_dia = np.array([d.hour for d in fechas])
        temp_base = 20 + 10 * np.sin(2 * np.pi * hora_del_dia / 24)
        temperatura = temp_base + np.random.normal(0, 2, horas)
        
        # Ocupación (0-100 personas)
        ocupacion = 30 + 40 * np.sin(2 * np.pi * hora_del_dia / 24) + np.random.normal(0, 5, horas)
        ocupacion = np.clip(ocupacion, 5, 100)
        
        # Consumo (kWh) - correlacionado con temp y ocupación
        # Consumo = base + coef_temp*temp + coef_ocup*ocupacion + ruido
        consumo = (
            10 +
            0.5 * temperatura +
            0.1 * ocupacion +
            np.random.normal(0, 1, horas)
        )
        
        df = pd.DataFrame({
            'fecha': fechas,
            'temperatura': temperatura,
            'ocupacion': ocupacion,
            'consumo': consumo
        })
        
        return df
    
    def agregar_caracteristicas_temporales(self, df):
        """Agrega características temporales."""
        df['hora'] = df['fecha'].dt.hour
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
        
        return df


class PredictorConsumoEnergia:
    """Predictor de consumo energético."""
    
    def __init__(self, seed=42):
        """Inicializa el predictor."""
        np.random.seed(seed)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.coeficientes = None
        self.metricas = {}
    
    def fit(self, X, y):
        """
        Entrena modelo de regresión lineal multivariada.
        
        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples,)
        """
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Agregar columna de bias
        X_scaled = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        # Regresión: β = (X^T X)^(-1) X^T y
        self.coeficientes = np.linalg.lstsq(X_scaled, y_scaled, rcond=None)[0]
        
        print(f"✅ Modelo entrenado")
        print(f"   Coeficientes: {self.coeficientes}")
    
    def predict(self, X):
        """Realiza predicciones."""
        X_scaled = self.scaler_X.transform(X)
        X_scaled = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        y_scaled = X_scaled @ self.coeficientes
        y = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        
        return y
    
    def evaluar(self, X, y):
        """Evalúa el modelo."""
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        self.metricas = {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        print(f"\n📊 Métricas:")
        print(f"   MSE:  {mse:.4f} kWh²")
        print(f"   RMSE: {rmse:.4f} kWh")
        print(f"   R²:   {r2:.4f}")
        
        return self.metricas
    
    def detectar_anomalias(self, X, y, umbral=2.0):
        """
        Detecta consumos anómalos.
        
        Args:
            X: Features
            y: Consumo real
            umbral: Número de std para considerar anomalía
        
        Returns:
            Índices de anomalías
        """
        y_pred = self.predict(X)
        residuos = np.abs(y - y_pred)
        
        media_residuos = residuos.mean()
        std_residuos = residuos.std()
        
        anomalias = np.where(residuos > media_residuos + umbral * std_residuos)[0]
        
        print(f"\n🚨 Anomalías detectadas: {len(anomalias)}")
        if len(anomalias) > 0:
            print(f"   Índices: {anomalias[:10].tolist()}")
            print(f"   Error medio: {residuos.mean():.2f} kWh")
        
        return anomalias


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("⚡ ANÁLISIS DE CONSUMO ENERGÉTICO - REGRESIÓN LINEAL MULTIVARIADA")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando datos de consumo...")
    generador = GeneradorDatosEnergia(seed=42)
    df = generador.generar_timeseries(dias=30)
    df = generador.agregar_caracteristicas_temporales(df)
    
    print(f"✅ Datos generados: {len(df)} registros")
    print(f"   Temperatura: [{df['temperatura'].min():.1f}, {df['temperatura'].max():.1f}]°C")
    print(f"   Ocupación: [{df['ocupacion'].min():.0f}, {df['ocupacion'].max():.0f}]")
    print(f"   Consumo: [{df['consumo'].min():.2f}, {df['consumo'].max():.2f}] kWh")
    
    # Paso 2: Preparar features
    print("\n[2] Preparando features...")
    features = ['temperatura', 'ocupacion', 'hora', 'es_fin_semana']
    X = df[features].values
    y = df['consumo'].values
    
    print(f"✅ Features: {features}")
    
    # Paso 3: Split
    print("\n[3] División train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Paso 4: Entrenar
    print("\n[4] Entrenando modelo...")
    modelo = PredictorConsumoEnergia()
    modelo.fit(X_train, y_train)
    
    # Paso 5: Evaluar
    print("\n[5] Evaluando...")
    modelo.evaluar(X_test, y_test)
    
    # Paso 6: Predicciones
    print("\n[6] Predicciones de ejemplo:")
    casos = [
        [15, 50, 9, 0],   # Mañana, laboral
        [25, 100, 14, 0], # Tarde, ocupado, laboral
        [10, 30, 2, 1],   # Madrugada, fin de semana
    ]
    
    for i, caso in enumerate(casos):
        pred = modelo.predict(np.array([caso]))[0]
        print(f"   Caso {i+1}: {pred:.2f} kWh")
    
    # Paso 7: Detectar anomalías
    print("\n[7] Detectando anomalías...")
    anomalias = modelo.detectar_anomalias(X_test, y_test, umbral=2.0)
    
    # Paso 8: Reporte
    print("\n[8] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'muestras': len(df),
        'metricas': modelo.metricas,
        'anomalias_detectadas': len(anomalias),
        'consumo_promedio': float(df['consumo'].mean()),
        'consumo_max': float(df['consumo'].max())
    }
    
    import json
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
