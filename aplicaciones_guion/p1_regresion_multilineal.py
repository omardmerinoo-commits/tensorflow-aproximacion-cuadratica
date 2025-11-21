#!/usr/bin/env python3
"""
P1: Predictor de Consumo de Energia - Regresion Multilineal
Predecir consumo basado en temperatura, humedad, ocupacion, hora
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json


class GeneradorDatosEnergia:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
    
    def generar_dataset(self, n_samples=500):
        """Generar datos de consumo de energia"""
        # Caracteristicas: temperatura, humedad, ocupacion, hora
        temperatura = np.random.uniform(15, 30, n_samples)
        humedad = np.random.uniform(30, 80, n_samples)
        ocupacion = np.random.uniform(0, 100, n_samples)
        hora = np.random.uniform(0, 24, n_samples)
        
        X = np.column_stack([temperatura, humedad, ocupacion, hora])
        
        # Consumo: mas alto con ocupacion, temperatura extrema
        y = 50 + 2*temperatura + 0.5*humedad + 0.8*ocupacion + 5*np.sin(hora*np.pi/12) + np.random.normal(0, 5, n_samples)
        y = np.maximum(y, 0)
        
        return {'X': X, 'y': y, 'features': ['Temperatura', 'Humedad', 'Ocupacion', 'Hora']}


class PredictorEnergia:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.coeficientes = None
    
    def fit(self, X, y):
        """Entrenar regresion multilineal"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        X_poly = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        self.coeficientes = np.linalg.lstsq(X_poly, y_scaled, rcond=None)[0]
        print("[+] Modelo P1 entrenado")
    
    def predict(self, X):
        """Prediccion"""
        X_scaled = self.scaler_X.transform(X)
        X_poly = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        y_scaled = X_poly @ self.coeficientes
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    
    def evaluar(self, X, y):
        """Evaluar"""
        y_pred = self.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return {'mae': float(mae), 'rmse': float(rmse)}


def main():
    print("\n" + "="*60)
    print("P1: PREDICTOR DE CONSUMO DE ENERGIA")
    print("="*60)
    
    generador = GeneradorDatosEnergia()
    datos = generador.generar_dataset(500)
    
    X_train, X_test, y_train, y_test = train_test_split(
        datos['X'], datos['y'], test_size=0.2, random_state=42)
    
    modelo = PredictorEnergia()
    modelo.fit(X_train, y_train)
    metricas = modelo.evaluar(X_test, y_test)
    
    print(f"MAE: {metricas['mae']:.2f} kWh")
    print(f"RMSE: {metricas['rmse']:.2f} kWh")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
