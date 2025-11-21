#!/usr/bin/env python3
"""
P2: Detector de Fraude - Clasificacion Logistica Binaria
Clasificar transacciones como legitimas (0) o fraudulentas (1)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class GeneradorTransacciones:
    def __init__(self, seed=42, fraud_rate=0.05):
        np.random.seed(seed)
        self.fraud_rate = fraud_rate
    
    def generar_dataset(self, n_samples=1000):
        """Generar transacciones legitimas y fraudulentas"""
        n_fraud = int(n_samples * self.fraud_rate)
        n_legit = n_samples - n_fraud
        
        # Legitimas: valores normales
        X_legit = np.random.normal([100, 50, 30], [50, 20, 10], (n_legit, 3))
        y_legit = np.zeros(n_legit)
        
        # Fraudulentas: valores anormales
        X_fraud = np.random.normal([300, 80, 70], [100, 40, 20], (n_fraud, 3))
        y_fraud = np.ones(n_fraud)
        
        X = np.vstack([X_legit, X_fraud])
        y = np.concatenate([y_legit, y_fraud])
        X = np.abs(X)
        
        idx = np.random.permutation(len(X))
        return {'X': X[idx], 'y': y[idx], 'features': ['Monto', 'Frecuencia', 'Riesgo']}


class DetectorFraude:
    def __init__(self, learning_rate=0.01, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.scaler = StandardScaler()
        self.pesos = None
        self.bias = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """Entrenar regresion logistica"""
        X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        
        self.pesos = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            z = np.dot(X, self.pesos) + self.bias
            pred = self.sigmoid(z)
            
            dw = (1 / n_samples) * np.dot(X.T, (pred - y))
            db = (1 / n_samples) * np.sum(pred - y)
            
            self.pesos -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        print("[+] Modelo P2 entrenado")
    
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        z = np.dot(X, self.pesos) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def evaluar(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        return {
            'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec),
            'f1': float(f1), 'cm': cm.tolist()
        }


def main():
    print("\n" + "="*60)
    print("P2: DETECTOR DE FRAUDE")
    print("="*60)
    
    generador = GeneradorTransacciones(fraud_rate=0.05)
    datos = generador.generar_dataset(1000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        datos['X'], datos['y'], test_size=0.2, random_state=42, stratify=datos['y'])
    
    detector = DetectorFraude(learning_rate=0.01, iterations=1000)
    detector.fit(X_train, y_train)
    metricas = detector.evaluar(X_test, y_test)
    
    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"F1-Score: {metricas['f1']:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
