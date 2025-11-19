"""
Aplicación: Clasificador de Diagnóstico Médico
==============================================

Caso de uso real: Sistema de apoyo al diagnóstico usando árboles de decisión.

Características:
- Análisis de síntomas
- Predicción de enfermedad
- Importancia de características
- Árbol interpretable

Autor: Proyecto TensorFlow
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json


class GeneradorDatosMedicos:
    """Generador de datos médicos sintéticos."""
    
    def __init__(self, seed=42):
        """Inicializa el generador."""
        np.random.seed(seed)
        self.seed = seed
        
        # Diccionarios de síntomas y diagnósticos
        self.sintomas = [
            'Fiebre', 'Tos', 'Dolor_garganta', 'Fatiga',
            'Congestion', 'Dolor_cabeza', 'Estornudos'
        ]
        
        self.diagnosticos = ['Resfriado', 'Gripe', 'Alergia', 'Bronquitis']
    
    def generar_dataset(self, n_samples=300):
        """
        Genera dataset de síntomas y diagnósticos.
        
        Args:
            n_samples: Número de muestras
        
        Returns:
            X: Síntomas (0-3 intensidad), y: Diagnósticos
        """
        X = []
        y = []
        
        for _ in range(n_samples):
            diagnostico = np.random.choice(self.diagnosticos)
            
            # Generar síntomas basados en diagnóstico
            if diagnostico == 'Resfriado':
                sintomas = [
                    np.random.randint(0, 2),  # Fiebre (rara)
                    np.random.randint(1, 4),  # Tos (común)
                    np.random.randint(1, 3),  # Dolor garganta
                    np.random.randint(1, 3),  # Fatiga
                    np.random.randint(1, 4),  # Congestión
                    np.random.randint(0, 2),  # Dolor cabeza (raro)
                    np.random.randint(1, 3)   # Estornudos
                ]
            elif diagnostico == 'Gripe':
                sintomas = [
                    np.random.randint(1, 4),  # Fiebre (común)
                    np.random.randint(1, 4),  # Tos
                    np.random.randint(0, 2),  # Dolor garganta (raro)
                    np.random.randint(2, 4),  # Fatiga (severa)
                    np.random.randint(0, 2),  # Congestión (rara)
                    np.random.randint(1, 4),  # Dolor cabeza
                    np.random.randint(0, 2)   # Estornudos (raro)
                ]
            elif diagnostico == 'Alergia':
                sintomas = [
                    np.random.randint(0, 1),  # Fiebre (no)
                    np.random.randint(0, 2),  # Tos (rara)
                    np.random.randint(0, 2),  # Dolor garganta
                    np.random.randint(0, 2),  # Fatiga (rara)
                    np.random.randint(1, 4),  # Congestión (común)
                    np.random.randint(0, 1),  # Dolor cabeza (no)
                    np.random.randint(2, 4)   # Estornudos (muy común)
                ]
            else:  # Bronquitis
                sintomas = [
                    np.random.randint(1, 3),  # Fiebre (a veces)
                    np.random.randint(2, 4),  # Tos (severa)
                    np.random.randint(0, 2),  # Dolor garganta
                    np.random.randint(1, 3),  # Fatiga
                    np.random.randint(1, 3),  # Congestión
                    np.random.randint(0, 2),  # Dolor cabeza
                    np.random.randint(0, 2)   # Estornudos (raro)
                ]
            
            X.append(sintomas)
            y.append(diagnostico)
        
        return {
            'X': np.array(X),
            'y': np.array(y),
            'features': self.sintomas,
            'clases': self.diagnosticos
        }


class ClasificadorDiagnostico:
    """Clasificador de diagnóstico médico."""
    
    def __init__(self, max_depth=5, seed=42):
        """Inicializa el clasificador."""
        self.max_depth = max_depth
        self.arbol = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=seed,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.label_encoder = LabelEncoder()
        self.features = None
        self.clases = None
        self.metricas = {}
    
    def fit(self, X, y, features=None, clases=None):
        """
        Entrena el árbol de decisión.
        
        Args:
            X: Features (síntomas)
            y: Targets (diagnósticos)
            features: Nombres de características
            clases: Nombres de clases
        """
        self.features = features
        self.clases = clases
        
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Entrenar
        self.arbol.fit(X, y_encoded)
        
        print(f"✅ Árbol entrenado")
        print(f"   Max depth: {self.arbol.get_depth()}")
        print(f"   Nodos: {self.arbol.tree_.node_count}")
    
    def predict(self, X):
        """Predice diagnósticos."""
        y_pred_encoded = self.arbol.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """Predice probabilidades."""
        return self.arbol.predict_proba(X)
    
    def evaluar(self, X, y):
        """Evalúa el modelo."""
        y_pred = self.predict(X)
        
        acc = accuracy_score(y, y_pred)
        
        self.metricas = {
            'accuracy': float(acc)
        }
        
        print(f"\n📊 Métricas de evaluación:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"\n   Reporte detallado:")
        print(classification_report(y, y_pred))
        
        cm = confusion_matrix(y, self.label_encoder.transform(y_pred))
        print(f"\n   Matriz de confusión:")
        print(cm)
        
        return self.metricas
    
    def obtener_importancia(self):
        """Obtiene importancia de características."""
        importancia = self.arbol.feature_importances_
        
        indices = np.argsort(importancia)[::-1]
        
        print(f"\n📊 Importancia de características:")
        for i in range(len(self.features)):
            idx = indices[i]
            if self.features:
                print(f"   {i+1}. {self.features[idx]:20s}: {importancia[idx]:.4f}")
            else:
                print(f"   {i+1}. Feature {idx:2d}: {importancia[idx]:.4f}")
        
        return importancia
    
    def diagnosticar_paciente(self, sintomas_intensidad):
        """
        Realiza diagnóstico para un paciente.
        
        Args:
            sintomas_intensidad: Array [fiebre, tos, dolor_garganta, ...]
        
        Returns:
            Dict con diagnóstico y confianza
        """
        X = np.array([sintomas_intensidad])
        diagnostico = self.predict(X)[0]
        probabilidades = self.predict_proba(X)[0]
        
        max_prob = probabilidades.max()
        
        return {
            'diagnostico': diagnostico,
            'confianza': float(max_prob),
            'probabilidades': {
                self.label_encoder.classes_[i]: float(p)
                for i, p in enumerate(probabilidades)
            }
        }


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("🏥 CLASIFICADOR DE DIAGNÓSTICO MÉDICO - ÁRRBOLES DE DECISIÓN")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando datos médicos...")
    generador = GeneradorDatosMedicos(seed=42)
    datos = generador.generar_dataset(n_samples=300)
    
    X = datos['X']
    y = datos['y']
    features = datos['features']
    clases = datos['clases']
    
    print(f"✅ Dataset generado: {len(X)} muestras")
    print(f"   Características: {features}")
    print(f"   Diagnósticos: {clases}")
    
    # Contar por clase
    for clase in clases:
        count = (y == clase).sum()
        print(f"     - {clase:15s}: {count:3d} ({100*count/len(y):5.1f}%)")
    
    # Paso 2: Split
    print("\n[2] División train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Paso 3: Entrenar
    print("\n[3] Entrenando árbol de decisión...")
    clasificador = ClasificadorDiagnostico(max_depth=5)
    clasificador.fit(X_train, y_train, features=features, clases=clases)
    
    # Paso 4: Evaluar
    print("\n[4] Evaluando en test set...")
    clasificador.evaluar(X_test, y_test)
    
    # Paso 5: Importancia
    print("\n[5] Analizando importancia...")
    clasificador.obtener_importancia()
    
    # Paso 6: Diagnósticos individuales
    print("\n[6] Diagnosticando pacientes:")
    casos = [
        [0, 2, 1, 1, 3, 0, 1],   # Parece resfriado
        [2, 1, 0, 3, 1, 2, 0],   # Parece gripe
        [0, 0, 0, 1, 3, 0, 3],   # Parece alergia
        [1, 3, 1, 1, 2, 0, 1],   # Parece bronquitis
    ]
    
    for i, sintomas in enumerate(casos):
        resultado = clasificador.diagnosticar_paciente(sintomas)
        print(f"\n   Paciente {i+1}:")
        print(f"     Síntomas: {sintomas}")
        print(f"     → Diagnóstico: {resultado['diagnostico']} ({resultado['confianza']:.2%} confianza)")
        print(f"       Probabilidades: {resultado['probabilidades']}")
    
    # Paso 7: Visualizar árbol
    print("\n[7] Generando visualización...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        clasificador.arbol,
        feature_names=features,
        class_names=clases,
        filled=True,
        rounded=True
    )
    plt.title("Árbol de Decisión para Diagnóstico Médico")
    plt.tight_layout()
    plt.savefig(output_dir / f"arbol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=100)
    plt.close()
    
    print(f"✅ Árbol visualizado")
    
    # Paso 8: Reporte
    print("\n[8] Generando reporte...")
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'Árbol de Decisión',
        'muestras': len(X),
        'metricas': clasificador.metricas,
        'configuracion': {
            'max_depth': 5,
            'min_samples_split': 5
        }
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
