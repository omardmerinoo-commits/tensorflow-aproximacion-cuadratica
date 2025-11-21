#!/usr/bin/env python3
"""
P0: Predictor de Precios de Casas - Regresion Cuadratica
Predecir precio basado en superficie usando relacion y = base + coef1*X + coef2*X^2
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from datetime import datetime


class GeneradorDatosCasas:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
    
    def generar_dataset(self, n_samples=200):
        """Genera datos sinteticos de precios de casas"""
        X = np.random.uniform(50, 500, n_samples)
        
        base = 50000
        coef1 = 1000
        coef2 = 5
        noise = np.random.normal(0, 10000, n_samples)
        
        y = base + coef1 * X + coef2 * (X ** 2) + noise
        y = np.maximum(y, 30000)
        
        return {
            'X': X.reshape(-1, 1),
            'y': y,
            'features': ['Superficie (m2)']
        }


class PredictorPreciosCasas:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.coeficientes = None
        self.metricas = {}
    
    def fit(self, X, y):
        """Entrenar modelo de regresion cuadratica"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        X_poly = np.column_stack([
            np.ones(len(X_scaled)),
            X_scaled,
            X_scaled ** 2
        ])
        
        self.coeficientes = np.linalg.lstsq(X_poly, y_scaled, rcond=None)[0]
        print("[+] Modelo P0 entrenado")
    
    def predict(self, X):
        """Prediccion"""
        X_scaled = self.scaler_X.transform(X)
        X_poly = np.column_stack([np.ones(len(X_scaled)), X_scaled, X_scaled ** 2])
        y_scaled = X_poly @ self.coeficientes
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    
    def evaluar(self, X, y):
        """Evaluar modelo"""
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        self.metricas = {'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
        return self.metricas


def main():
    print("\n" + "="*60)
    print("P0: PREDICTOR DE PRECIOS DE CASAS")
    print("="*60)
    
    generador = GeneradorDatosCasas()
    datos = generador.generar_dataset(200)
    X, y = datos['X'], datos['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = PredictorPreciosCasas()
    modelo.fit(X_train, y_train)
    metricas = modelo.evaluar(X_test, y_test)
    
    print(f"RMSE: ${metricas['rmse']:,.0f}")
    print(f"MAE: ${metricas['mae']:,.0f}")
    print(f"R2: {metricas['r2']:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
