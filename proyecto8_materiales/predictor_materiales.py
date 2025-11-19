"""
Proyecto 8: Predictor de Propiedades de Materiales con Regresi\u00f3n Multivariada
================================================================================

Sistema para predecir propiedades f\u00edsicas de materiales basado en
composici\u00f3n elemental y par\u00e1metros estructurales.

Propiedades predichas:
- Densidad (g/cm³): Masa por unidad de volumen
- Dureza (Mohs): Resistencia a rayado
- Punto de fusi\u00f3n (K): Temperatura de cambio de fase s\u00f3lido-l\u00edquido

Caracter\u00edsticas:
- Generaci\u00f3n sint\u00e9tica realista de datos de materiales
- Preprocesamiento con normalizaci\u00f3n
- Modelos: Regresi\u00f3n lineal, MLP, ensemble
- Evaluaci\u00f3n con m\u00e9tricas: MSE, MAE, R², RMSE
- Validaci\u00f3n cruzada y detecci\u00f3n de outliers

Teor\u00eda:
La regresi\u00f3n multivariada busca encontrar una funci\u00f3n f: X \u2192 Y
que mapee features a propiedades. Con ruido y relaciones no-lineales,
redes neuronales capturan dependencias complejas entre elementos.

"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DatosMateria les:
    """Contenedor para datos de materiales"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    nombres_propiedades: List[str]
    nombres_features: List[str]
    
    def info(self) -> str:
        return (f"Materiales Dataset: Train {self.X_train.shape}, "
                f"Val {self.X_val.shape}, Test {self.X_test.shape}, "
                f"Features {self.X_train.shape[1]}, Propiedades {self.y_train.shape[1]}")


class GeneradorMateriales:
    """Generador de datos sint\u00e9ticos para materiales"""
    
    # Elementos simb\u00f3licos y sus propiedades
    ELEMENTOS = {
        'Fe': {'masa_atomica': 55.845, 'radio_atomico': 126},
        'Cu': {'masa_atomica': 63.546, 'radio_atomico': 135},
        'Al': {'masa_atomica': 26.982, 'radio_atomico': 143},
        'Si': {'masa_atomica': 28.086, 'radio_atomico': 111},
        'C': {'masa_atomica': 12.011, 'radio_atomico': 77},
        'Ni': {'masa_atomica': 58.693, 'radio_atomico': 124},
        'Ti': {'masa_atomica': 47.867, 'radio_atomico': 147},
        'Zn': {'masa_atomica': 65.380, 'radio_atomico': 134},
    }
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def _generar_composicion(self, n_elementos: int = 3) -> np.ndarray:
        """Genera composici\u00f3n elemental aleatoria"""
        n_elem_selec = np.random.randint(n_elementos - 1, n_elementos + 1)
        elementos_selec = np.random.choice(list(self.ELEMENTOS.keys()),
                                          n_elem_selec, replace=False)
        
        composicion = np.zeros(len(self.ELEMENTOS))
        for i, elem in enumerate(self.ELEMENTOS.keys()):
            if elem in elementos_selec:
                composicion[i] = np.random.uniform(0.1, 0.9)
        
        # Normalizar a proporciones
        composicion = composicion / (composicion.sum() + 1e-8)
        return composicion
    
    def _calcular_densidad(self, composicion: np.ndarray,
                          porosidad: float) -> float:
        """Calcula densidad te\u00f3rica"""
        densidades_elemento = {
            'Fe': 7.87, 'Cu': 8.96, 'Al': 2.70,
            'Si': 2.33, 'C': 2.26, 'Ni': 8.90, 'Ti': 4.51, 'Zn': 7.14
        }
        
        densidad_teorica = sum(
            composicion[i] * densidades_elemento[elem]
            for i, elem in enumerate(self.ELEMENTOS.keys())
        )
        
        # Aplicar porosidad
        densidad = densidad_teorica * (1 - porosidad)
        return densidad
    
    def _calcular_dureza(self, composicion: np.ndarray,
                        temperatura_procesamiento: float) -> float:
        """Calcula dureza Mohs aprox"""
        # Dureza base por elemento
        dureza_elemento = {
            'Fe': 4.0, 'Cu': 2.5, 'Al': 2.75,
            'Si': 7.0, 'C': 10.0, 'Ni': 4.0, 'Ti': 6.0, 'Zn': 2.5
        }
        
        dureza = sum(
            composicion[i] * dureza_elemento[elem]
            for i, elem in enumerate(self.ELEMENTOS.keys())
        )
        
        # Temperatura aumenta dureza (endurecimiento por precipitaci\u00f3n)
        factor_temp = 1 + 0.001 * np.sqrt(max(0, temperatura_procesamiento))
        dureza *= factor_temp
        
        # Ruido
        dureza += np.random.normal(0, 0.3)
        return np.clip(dureza, 1, 10)
    
    def _calcular_punto_fusion(self, composicion: np.ndarray) -> float:
        """Calcula punto de fusi\u00f3n aprox (K)"""
        punto_fusion_elemento = {
            'Fe': 1811, 'Cu': 1358, 'Al': 933,
            'Si': 1687, 'C': 3823, 'Ni': 1728, 'Ti': 1941, 'Zn': 693
        }
        
        punto_fusion = sum(
            composicion[i] * punto_fusion_elemento[elem]
            for i, elem in enumerate(self.ELEMENTOS.keys())
        )
        
        # Ruido
        punto_fusion += np.random.normal(0, 50)
        return max(300, punto_fusion)
    
    def generar(self, n_muestras: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Genera dataset de materiales"""
        X = []
        y = []
        
        for _ in range(n_muestras):
            # Composici\u00f3n elemental (8 features)
            composicion = self._generar_composicion()
            
            # Par\u00e1metros estructurales (3 features)
            porosidad = np.random.uniform(0, 0.3)
            tamano_grano = np.random.uniform(1, 1000)  # microm
            temperatura_procesamiento = np.random.uniform(300, 1200)  # K
            
            # Features
            features = np.concatenate([
                composicion,
                [porosidad, tamano_grano, temperatura_procesamiento]
            ])
            
            # Propiedades (targets)
            densidad = self._calcular_densidad(composicion, porosidad)
            dureza = self._calcular_dureza(composicion, temperatura_procesamiento)
            punto_fusion = self._calcular_punto_fusion(composicion)
            
            X.append(features)
            y.append([densidad, dureza, punto_fusion])
        
        return np.array(X), np.array(y)
    
    def generar_dataset(self, n_muestras: int = 500,
                       split: Tuple[float, float, float] = (0.6, 0.2, 0.2)
                       ) -> 'DatosMateria les':
        """Genera dataset completo con splits"""
        X, y = self.generar(n_muestras)
        
        # Shuffle
        indices = np.random.permutation(len(y))
        X, y = X[indices], y[indices]
        
        # Split
        train_ratio, val_ratio, test_ratio = split
        n_train = int(len(y) * train_ratio)
        n_val = int(len(y) * val_ratio)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        nombres_propiedades = ['Densidad (g/cm³)', 'Dureza (Mohs)', 'P. Fusi\u00f3n (K)']
        nombres_features = (list(self.ELEMENTOS.keys()) + 
                          ['Porosidad', 'Tamano_grano', 'Temp_proc'])
        
        return DatosMateria les(
            X_train, y_train, X_val, y_val, X_test, y_test,
            nombres_propiedades, nombres_features
        )


class PredictorMateriales:
    """Predictor de propiedades de materiales"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.entrenado = False
        self.n_salidas = None
    
    def _normalizar(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normaliza features"""
        if fit:
            return self.scaler_X.fit_transform(X)
        return self.scaler_X.transform(X)
    
    def _normalizar_salida(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normaliza targets"""
        if fit:
            return self.scaler_y.fit_transform(y)
        return self.scaler_y.transform(y)
    
    def _desnormalizar_salida(self, y_norm: np.ndarray) -> np.ndarray:
        """Desnormaliza predicciones"""
        return self.scaler_y.inverse_transform(y_norm)
    
    def construir_mlp(self, capas_ocultas: List[int] = None,
                     n_salidas: int = 3) -> models.Model:
        """Construye MLP para regresi\u00f3n multivariada"""
        if capas_ocultas is None:
            capas_ocultas = [256, 128, 64]
        
        modelo = models.Sequential()
        modelo.add(layers.Input(shape=(11,)))  # 11 features
        
        # Capas ocultas
        for units in capas_ocultas:
            modelo.add(layers.Dense(units, activation='relu'))
            modelo.add(layers.BatchNormalization())
            modelo.add(layers.Dropout(0.3))
        
        # Capa de salida
        modelo.add(layers.Dense(n_salidas))
        
        return modelo
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 50, verbose: int = 1) -> Dict:
        """Entrena el modelo"""
        X_train_norm = self._normalizar(X_train, fit=True)
        X_val_norm = self._normalizar(X_val)
        y_train_norm = self._normalizar_salida(y_train, fit=True)
        y_val_norm = self._normalizar_salida(y_val)
        
        self.n_salidas = y_train.shape[1]
        self.modelo = self.construir_mlp(n_salidas=self.n_salidas)
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
            )
        ]
        
        hist = self.modelo.fit(
            X_train_norm, y_train_norm,
            validation_data=(X_val_norm, y_val_norm),
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            batch_size=16
        )
        
        self.entrenado = True
        return hist.history
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Eval\u00faa el modelo"""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        X_test_norm = self._normalizar(X_test)
        y_test_norm = self._normalizar_salida(y_test)
        
        y_pred_norm = self.modelo.predict(X_test_norm, verbose=0)
        y_pred = self._desnormalizar_salida(y_pred_norm)
        
        loss = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(loss)
        
        return {
            'mse': loss,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'predicciones': y_pred,
            'residuos': y_test - y_pred
        }
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones"""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        X_norm = self._normalizar(X)
        y_pred_norm = self.modelo.predict(X_norm, verbose=0)
        return self._desnormalizar_salida(y_pred_norm)
    
    def guardar(self, ruta: str):
        """Guarda el modelo"""
        self.modelo.save(f"{ruta}_modelo.h5")
        with open(f"{ruta}_scaler_X.pkl", 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open(f"{ruta}_scaler_y.pkl", 'wb') as f:
            pickle.dump(self.scaler_y, f)
    
    @staticmethod
    def cargar(ruta: str) -> 'PredictorMateriales':
        """Carga un modelo guardado"""
        predictor = PredictorMateriales()
        predictor.modelo = keras.models.load_model(f"{ruta}_modelo.h5")
        with open(f"{ruta}_scaler_X.pkl", 'rb') as f:
            predictor.scaler_X = pickle.load(f)
        with open(f"{ruta}_scaler_y.pkl", 'rb') as f:
            predictor.scaler_y = pickle.load(f)
        predictor.entrenado = True
        return predictor


def demo():
    """Demostraci\u00f3n completa"""
    print("="*70)
    print("PREDICTOR DE PROPIEDADES DE MATERIALES - DEMOSTRACIÓN")
    print("="*70)
    
    # 1. Generar datos
    print("\n[1] Generando datos de materiales...")
    generador = GeneradorMateriales()
    datos = generador.generar_dataset(n_muestras=500)
    print(f"✓ {datos.info()}")
    print(f"  Propiedades: {', '.join(datos.nombres_propiedades)}")
    
    # 2. Entrenar
    print("\n[2] Entrenando modelo...")
    predictor = PredictorMateriales()
    predictor.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20, verbose=0
    )
    
    # 3. Evaluar
    print("\n[3] Evaluando...")
    metricas = predictor.evaluar(datos.X_test, datos.y_test)
    print(f"✓ Métricas por propiedad:")
    for i, prop in enumerate(datos.nombres_propiedades):
        print(f"  {prop}:")
        print(f"    R²: {metricas['r2_score'][i]:.4f}")
        print(f"    RMSE: {metricas['rmse'][i]:.4f}")
    
    # 4. Predecir
    print("\n[4] Predicciones...")
    predicciones = predictor.predecir(datos.X_test[:5])
    print("Comparaci\u00f3n de predicciones:")
    for i in range(5):
        print(f"\n  Muestra {i}:")
        for j, prop in enumerate(datos.nombres_propiedades):
            print(f"    {prop}: Real={datos.y_test[i, j]:.2f}, Pred={predicciones[i, j]:.2f}")
    
    print("\n✓ Demostración completada")


if __name__ == '__main__':
    demo()
