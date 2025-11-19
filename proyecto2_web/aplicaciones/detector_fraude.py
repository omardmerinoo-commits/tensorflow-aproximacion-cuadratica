"""
Aplicación: Detección de Fraude en Transacciones
================================================

Caso de uso real: Sistema de detección de transacciones fraudulentas usando clasificación logística.

Características:
- Análisis de patrones de transacción
- Detección de anomalías
- Alertas en tiempo real
- Matriz de confusión y ROC

Autor: Proyecto TensorFlow
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
from datetime import datetime
from pathlib import Path
import json


class GeneradorTransacciones:
    """Generador de datos de transacciones."""
    
    def __init__(self, seed=42, fraud_rate=0.05):
        """Inicializa el generador."""
        np.random.seed(seed)
        self.seed = seed
        self.fraud_rate = fraud_rate
    
    def generar_dataset(self, n_samples=1000):
        """
        Genera dataset de transacciones.
        
        Args:
            n_samples: Número de muestras
        
        Returns:
            X: Features, y: Etiquetas (0=legítimo, 1=fraude)
        """
        n_fraud = int(n_samples * self.fraud_rate)
        n_legit = n_samples - n_fraud
        
        # Transacciones legítimas
        X_legit = np.random.normal([100, 50, 30], [50, 20, 10], (n_legit, 3))
        y_legit = np.zeros(n_legit)
        
        # Transacciones fraudulentas (distribución diferente)
        X_fraud = np.random.normal([300, 80, 70], [100, 40, 20], (n_fraud, 3))
        y_fraud = np.ones(n_fraud)
        
        # Combinar
        X = np.vstack([X_legit, X_fraud])
        y = np.concatenate([y_legit, y_fraud])
        
        # Asegurar valores positivos
        X = np.abs(X)
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        return {
            'X': X,
            'y': y,
            'features': ['Monto ($)', 'Frecuencia (compras/mes)', 'Riesgo (0-100)']
        }


class DetectorFraude:
    """Detector de fraude basado en regresión logística."""
    
    def __init__(self, learning_rate=0.01, iterations=100, seed=42):
        """Inicializa el detector."""
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.scaler = StandardScaler()
        self.pesos = None
        self.bias = 0
        self.metricas = {}
    
    def sigmoid(self, z):
        """Función sigmoid."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """
        Entrena modelo de regresión logística.
        
        Args:
            X: Features (n_samples, n_features)
            y: Targets binarios (0 o 1)
        """
        X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        
        # Inicializar pesos
        self.pesos = np.zeros(n_features)
        self.bias = 0
        
        # Descenso de gradiente
        for _ in range(self.iterations):
            # Predicciones
            z = np.dot(X, self.pesos) + self.bias
            predicciones = self.sigmoid(z)
            
            # Gradientes
            dw = (1 / n_samples) * np.dot(X.T, (predicciones - y))
            db = (1 / n_samples) * np.sum(predicciones - y)
            
            # Actualizar pesos
            self.pesos -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        print(f"✅ Modelo entrenado ({self.iterations} iteraciones)")
    
    def predict_proba(self, X):
        """Predice probabilidades."""
        X = self.scaler.transform(X)
        z = np.dot(X, self.pesos) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predice clases."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluar(self, X, y, threshold=0.5):
        """Evalúa el modelo."""
        y_pred = self.predict(X, threshold)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        cm = confusion_matrix(y, y_pred)
        
        self.metricas = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        print(f"\n📊 Métricas de evaluación:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1:        {f1:.4f}")
        print(f"\n   Matriz de confusión:")
        print(f"   [[{cm[0,0]:5d} {cm[0,1]:5d}]  (TN, FP)")
        print(f"    [{cm[1,0]:5d} {cm[1,1]:5d}]] (FN, TP)")
        
        return self.metricas
    
    def obtener_roc(self, X, y):
        """Obtiene curva ROC."""
        y_proba = self.predict_proba(X)
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        auc_score = auc(fpr, tpr)
        
        return {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}
    
    def reportar_transaccion(self, monto, frecuencia, riesgo, threshold=0.5):
        """
        Analiza una transacción individual.
        
        Args:
            monto: Monto en $
            frecuencia: Compras por mes
            riesgo: Score de riesgo (0-100)
        
        Returns:
            Dict con resultado
        """
        X = np.array([[monto, frecuencia, riesgo]])
        proba = self.predict_proba(X)[0]
        pred = self.predict(X, threshold)[0]
        
        resultado = 'FRAUDE' if pred == 1 else 'LEGÍTIMO'
        
        return {
            'transaccion': {'monto': monto, 'frecuencia': frecuencia, 'riesgo': riesgo},
            'probabilidad_fraude': float(proba),
            'prediccion': resultado,
            'confianza': float(max(proba, 1 - proba))
        }


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("🔐 DETECTOR DE FRAUDE - CLASIFICACIÓN LOGÍSTICA")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando datos de transacciones...")
    generador = GeneradorTransacciones(seed=42, fraud_rate=0.05)
    datos = generador.generar_dataset(n_samples=1000)
    
    X = datos['X']
    y = datos['y']
    
    print(f"✅ Dataset generado: {len(X)} muestras")
    print(f"   Legítimas: {int((y==0).sum())} ({100*(y==0).sum()/len(y):.1f}%)")
    print(f"   Fraude:    {int((y==1).sum())} ({100*(y==1).sum()/len(y):.1f}%)")
    
    # Paso 2: Split
    print("\n[2] División train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Paso 3: Entrenar
    print("\n[3] Entrenando detector...")
    detector = DetectorFraude(learning_rate=0.01, iterations=1000)
    detector.fit(X_train, y_train)
    
    # Paso 4: Evaluar
    print("\n[4] Evaluando en test set...")
    detector.evaluar(X_test, y_test, threshold=0.5)
    
    # Paso 5: Análisis ROC
    print("\n[5] Analizando curva ROC...")
    roc = detector.obtener_roc(X_test, y_test)
    print(f"✅ AUC Score: {roc['auc']:.4f}")
    
    # Paso 6: Análisis de transacciones individuales
    print("\n[6] Analizando transacciones individuales:")
    casos = [
        (100, 50, 20),   # Típica legítima
        (500, 80, 75),   # Sospechosa
        (50, 10, 5),     # Muy legítima
        (1000, 1, 90),   # Muy sospechosa
    ]
    
    for i, (monto, freq, riesgo) in enumerate(casos):
        resultado = detector.reportar_transaccion(monto, freq, riesgo)
        trans = resultado['transaccion']
        print(f"\n   Transacción {i+1}:")
        print(f"     Monto: ${trans['monto']}")
        print(f"     Frecuencia: {trans['frecuencia']}")
        print(f"     Riesgo: {trans['riesgo']}")
        print(f"     → {resultado['prediccion']} (prob: {resultado['probabilidad_fraude']:.2%})")
    
    # Paso 7: Reporte
    print("\n[7] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'Regresión Logística',
        'muestras': len(X),
        'metricas': detector.metricas,
        'auc_score': roc['auc'],
        'configuracion': {
            'learning_rate': 0.01,
            'iterations': 1000,
            'threshold': 0.5
        }
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte guardado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
