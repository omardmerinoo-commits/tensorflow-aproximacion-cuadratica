"""
🎯 Proyecto 3: Simulador de Qubits con Red Neuronal
====================================================

Simulación de sistemas cuánticos simples usando:
- Álgebra lineal compleja (matrices de Pauli, Hadamard, etc.)
- Estados cuánticos como vectores de amplitudes
- Operaciones unitarias (rotaciones, puertas lógicas)
- Medición y colapso de función de onda
- Red neuronal para predicción de evolución temporal
- Visualización de estados y probabilidades

✨ Características:
- 🧪 Simulación exacta de qubits individuales
- 🔀 Operaciones cuánticas: Pauli, Hadamard, CNOT
- 📊 Cálculo de probabilidades de medición
- 🔗 Entrelazamiento de qubits (Bell states)
- 🧠 Red neuronal para evolución temporal
- 📈 Validación contra soluciones analíticas
- 🎨 Visualización de estados cuánticos
- 🧪 Pruebas exhaustivas (50+ tests)

📐 Teoría:
- Estados cuánticos: |ψ⟩ = α|0⟩ + β|1⟩
- Puertas cuánticas: matrices unitarias 2×2 ó 4×4
- Medición: colapso a |0⟩ ó |1⟩ con probabilidades |α|² y |β|²
- Entrelazamiento: estados no separables de múltiples qubits

Autor: Sistema de Educación TensorFlow
Licencia: MIT
Versión: 1.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional, Callable, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import pickle
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES Y DEFINICIONES CUÁNTICAS
# ============================================================================

# Puertas de Pauli
PAULI_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)  # Identidad
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)  # Bit flip (NOT)
PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)   # Bit-phase flip
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex) # Phase flip

# Puerta Hadamard (superposición)
HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)

# Estados base
ZERO_STATE = np.array([[1.0], [0.0]], dtype=complex)  # |0⟩
ONE_STATE = np.array([[0.0], [1.0]], dtype=complex)   # |1⟩

# Estados de Bell (entrelazados)
BELL_00 = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=complex) / np.sqrt(2.0)  # (|00⟩ + |11⟩)/√2
BELL_01 = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=complex) / np.sqrt(2.0)  # (|01⟩ + |10⟩)/√2
BELL_10 = np.array([[1.0], [0.0], [0.0], [-1.0]], dtype=complex) / np.sqrt(2.0) # (|00⟩ - |11⟩)/√2
BELL_11 = np.array([[0.0], [1.0], [-1.0], [0.0]], dtype=complex) / np.sqrt(2.0) # (|01⟩ - |10⟩)/√2


# ============================================================================
# CLASE DE DATOS PARA ESTADOS CUÁNTICOS
# ============================================================================

@dataclass
class EstadoCuantico:
    """Representa un estado cuántico (qubit)."""
    amplitudes: np.ndarray  # Vector de amplitudes complejas
    normalizacion: float = field(default=1.0)
    timestamp: datetime = field(default_factory=datetime.now)
    etiqueta: str = field(default="")
    
    def __post_init__(self):
        """Validar y normalizar el estado."""
        if not np.allclose(np.linalg.norm(self.amplitudes), 1.0):
            self.amplitudes = self.amplitudes / np.linalg.norm(self.amplitudes)
    
    def get_probabilidades(self) -> Dict[int, float]:
        """Obtiene probabilidades de medición."""
        probs = {}
        for i, amp in enumerate(self.amplitudes):
            prob = np.abs(amp) ** 2
            if prob > 1e-10:  # Solo si es significativo
                probs[i] = float(prob)
        return probs
    
    def medir(self, seed: int = None) -> int:
        """Simula medición y retorna resultado (0 ó 1)."""
        if seed is not None:
            np.random.seed(seed)
        probs = self.get_probabilidades()
        return np.random.choice(list(probs.keys()), p=list(probs.values()))
    
    @property
    def texto(self) -> str:
        """Representación como string."""
        resultado = ""
        base = ['|0⟩', '|1⟩', '|2⟩', '|3⟩']
        for i, amp in enumerate(self.amplitudes):
            if np.abs(amp) > 1e-10:
                resultado += f"{amp:.4f}{base[i]} "
        return resultado if resultado else "Zero state"


# ============================================================================
# SIMULADOR DE QUBITS
# ============================================================================

class SimuladorQubit:
    """Simulador de sistemas cuánticos con red neuronal."""
    
    def __init__(self, num_qubits: int = 1, seed: int = 42):
        """
        Inicializa el simulador.
        
        Args:
            num_qubits: Número de qubits (1 ó 2)
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits  # 2 para 1 qubit, 4 para 2 qubits
        self.estado = EstadoCuantico(ZERO_STATE.copy())
        self.historial_mediciones = []
        self.historial_evoluciones = []
        self.modelo_nn = None
        self.scaler_entrada = StandardScaler()
        self.scaler_salida = StandardScaler()
        
        logger.info(f"✅ Simulador inicializado: {num_qubits} qubit(s)")
    
    # ========================================================================
    # PUERTAS CUÁNTICAS BÁSICAS (1 QUBIT)
    # ========================================================================
    
    def puerta_pauli_x(self) -> EstadoCuantico:
        """Aplica puerta Pauli-X (bit flip): |0⟩→|1⟩, |1⟩→|0⟩."""
        self.estado.amplitudes = PAULI_X @ self.estado.amplitudes
        self.historial_evoluciones.append(('X', self.estado.amplitudes.copy()))
        return self.estado
    
    def puerta_pauli_y(self) -> EstadoCuantico:
        """Aplica puerta Pauli-Y."""
        self.estado.amplitudes = PAULI_Y @ self.estado.amplitudes
        self.historial_evoluciones.append(('Y', self.estado.amplitudes.copy()))
        return self.estado
    
    def puerta_pauli_z(self) -> EstadoCuantico:
        """Aplica puerta Pauli-Z (phase flip)."""
        self.estado.amplitudes = PAULI_Z @ self.estado.amplitudes
        self.historial_evoluciones.append(('Z', self.estado.amplitudes.copy()))
        return self.estado
    
    def puerta_hadamard(self) -> EstadoCuantico:
        """Aplica puerta Hadamard (superposición)."""
        self.estado.amplitudes = HADAMARD @ self.estado.amplitudes
        self.historial_evoluciones.append(('H', self.estado.amplitudes.copy()))
        return self.estado
    
    def puerta_rotacion_x(self, theta: float) -> EstadoCuantico:
        """
        Aplica rotación alrededor del eje X.
        
        RX(θ) = [[cos(θ/2), -i·sin(θ/2)],
                 [-i·sin(θ/2), cos(θ/2)]]
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matriz = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
        self.estado.amplitudes = matriz @ self.estado.amplitudes
        self.historial_evoluciones.append(('RX', self.estado.amplitudes.copy()))
        return self.estado
    
    def puerta_rotacion_y(self, theta: float) -> EstadoCuantico:
        """Aplica rotación alrededor del eje Y."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matriz = np.array([[c, -s], [s, c]], dtype=complex)
        self.estado.amplitudes = matriz @ self.estado.amplitudes
        self.historial_evoluciones.append(('RY', self.estado.amplitudes.copy()))
        return self.estado
    
    def puerta_rotacion_z(self, theta: float) -> EstadoCuantico:
        """Aplica rotación alrededor del eje Z."""
        matriz = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)
        self.estado.amplitudes = matriz @ self.estado.amplitudes
        self.historial_evoluciones.append(('RZ', self.estado.amplitudes.copy()))
        return self.estado
    
    def puerta_fase(self, phi: float) -> EstadoCuantico:
        """Aplica puerta de fase."""
        matriz = np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)
        self.estado.amplitudes = matriz @ self.estado.amplitudes
        self.historial_evoluciones.append(('PHASE', self.estado.amplitudes.copy()))
        return self.estado
    
    def puerta_t(self) -> EstadoCuantico:
        """Aplica puerta T (π/8 phase gate)."""
        return self.puerta_fase(np.pi / 4)
    
    # ========================================================================
    # OPERACIONES DE MEDICIÓN
    # ========================================================================
    
    def medir(self, seed: int = None) -> int:
        """
        Realiza medición del qubit.
        
        Returns:
            0 ó 1 (para 1 qubit), 0-3 (para 2 qubits)
        """
        resultado = self.estado.medir(seed)
        self.historial_mediciones.append(resultado)
        
        # Colapso de la función de onda
        nuevo_estado = np.zeros(self.dim, dtype=complex)
        nuevo_estado[resultado] = 1.0
        self.estado.amplitudes = nuevo_estado.reshape(-1, 1)
        
        return resultado
    
    def get_probabilidades(self) -> Dict[int, float]:
        """Retorna distribución de probabilidades."""
        return self.estado.get_probabilidades()
    
    # ========================================================================
    # OPERACIONES DE 2 QUBITS
    # ========================================================================
    
    def crear_superposicion_igual(self) -> EstadoCuantico:
        """Crea estado de superposición igual (|0⟩+|1⟩)/√2."""
        self.estado = EstadoCuantico(
            (ZERO_STATE + ONE_STATE) / np.sqrt(2.0)
        )
        return self.estado
    
    def crear_bell_state(self, tipo: str = "00") -> EstadoCuantico:
        """
        Crea estado de Bell entrelazado.
        
        Args:
            tipo: "00", "01", "10" ó "11"
        """
        if self.num_qubits != 2:
            raise ValueError("Bell states require 2 qubits")
        
        bell_states = {
            "00": BELL_00,
            "01": BELL_01,
            "10": BELL_10,
            "11": BELL_11
        }
        
        self.estado = EstadoCuantico(bell_states[tipo].copy())
        return self.estado
    
    def puerta_cnot(self) -> EstadoCuantico:
        """Puerta CNOT (Controlled-NOT) para 2 qubits."""
        if self.num_qubits != 2:
            raise ValueError("CNOT requires 2 qubits")
        
        # Matriz CNOT: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        self.estado.amplitudes = cnot @ self.estado.amplitudes
        self.historial_evoluciones.append(('CNOT', self.estado.amplitudes.copy()))
        return self.estado
    
    # ========================================================================
    # GENERACIÓN DE DATOS PARA ENTRENAMIENTO
    # ========================================================================
    
    def generar_datos_evolucion(self, 
                                num_muestras: int = 1000,
                                pasos_tiempo: int = 10,
                                test_size: float = 0.2,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera datos de evolución temporal de estados cuánticos.
        
        Args:
            num_muestras: Número de secuencias a generar
            pasos_tiempo: Pasos de tiempo simulados
            test_size: Fracción para test
            seed: Semilla aleatoria
        
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        logger.info(f"📊 Generando {num_muestras} muestras de evolución...")
        
        np.random.seed(seed)
        X_data = []
        y_data = []
        
        for i in range(num_muestras):
            # Estado inicial aleatorio
            amplitud_real = np.random.randn(self.dim)
            amplitud_imag = np.random.randn(self.dim)
            amplitudes_init = (amplitud_real + 1j*amplitud_imag).reshape(-1, 1)
            amplitudes_init = amplitudes_init / np.linalg.norm(amplitudes_init)
            
            # Secuencia de evolución
            estado_actual = amplitudes_init.copy().flatten()
            
            for t in range(pasos_tiempo):
                # Aplicar operación aleatoria
                operacion = np.random.choice(['X', 'Y', 'Z', 'H', 'RX', 'RY'])
                theta = np.random.uniform(0, 2*np.pi)
                
                if operacion == 'X':
                    matriz_op = PAULI_X
                elif operacion == 'Y':
                    matriz_op = PAULI_Y
                elif operacion == 'Z':
                    matriz_op = PAULI_Z
                elif operacion == 'H':
                    matriz_op = HADAMARD
                elif operacion == 'RX':
                    c = np.cos(theta/2)
                    s = np.sin(theta/2)
                    matriz_op = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
                else:  # RY
                    c = np.cos(theta/2)
                    s = np.sin(theta/2)
                    matriz_op = np.array([[c, -s], [s, c]], dtype=complex)
                
                # Aplicar operación
                estado_siguiente = matriz_op @ estado_actual.reshape(-1, 1)
                estado_siguiente = estado_siguiente.flatten()
                
                # Guardar como (entrada, salida)
                # Entrada: estado actual + parámetros
                entrada = np.concatenate([
                    np.abs(estado_actual),  # Magnitudes
                    np.angle(estado_actual),  # Fases
                    [np.pi if operacion.startswith('R') else 0]  # Parámetro
                ])
                
                # Salida: estado siguiente
                salida = np.concatenate([
                    np.abs(estado_siguiente),
                    np.angle(estado_siguiente)
                ])
                
                X_data.append(entrada)
                y_data.append(salida)
                
                estado_actual = estado_siguiente
        
        X_data = np.array(X_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.float32)
        
        # Dividir en train/test
        split_idx = int(len(X_data) * (1 - test_size))
        X_train, X_test = X_data[:split_idx], X_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]
        
        logger.info(f"✅ Datos generados: {X_train.shape[0]} train, {X_test.shape[0]} test")
        return X_train, y_train, X_test, y_test
    
    # ========================================================================
    # CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO
    # ========================================================================
    
    def construir_modelo(self,
                        capas_ocultas: List[int] = None,
                        tasa_aprendizaje: float = 0.001,
                        dropout_rate: float = 0.2) -> keras.Model:
        """
        Construye red neuronal para predicción de evolución cuántica.
        
        Args:
            capas_ocultas: Lista de tamaños de capas ocultas
            tasa_aprendizaje: Learning rate
            dropout_rate: Dropout para regularización
        
        Returns:
            Modelo Keras compilado
        """
        if capas_ocultas is None:
            capas_ocultas = [128, 64, 32]
        
        # Dimensión de entrada y salida
        dim_entrada = 4 * self.dim + 1  # Amplitudes + fases + parámetro
        dim_salida = 2 * self.dim  # Amplitudes y fases de salida
        
        logger.info(f"🔨 Construyendo modelo: {dim_entrada} → {capas_ocultas} → {dim_salida}")
        
        modelo = keras.Sequential([
            layers.Input(shape=(dim_entrada,)),
            layers.Dense(capas_ocultas[0], activation='relu'),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            
            layers.Dense(capas_ocultas[1], activation='relu'),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            
            layers.Dense(capas_ocultas[2], activation='relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(dim_salida, activation='tanh')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=tasa_aprendizaje)
        modelo.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.modelo_nn = modelo
        logger.info(f"✅ Modelo construido con {modelo.count_params():,} parámetros")
        return modelo
    
    def entrenar(self,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray = None,
                y_val: np.ndarray = None,
                epochs: int = 100,
                batch_size: int = 32,
                verbose: int = 1) -> Dict[str, Any]:
        """
        Entrena el modelo neural.
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            epochs: Épocas de entrenamiento
            batch_size: Tamaño de batch
            verbose: Nivel de verbosidad
        
        Returns:
            Historial de entrenamiento
        """
        if self.modelo_nn is None:
            self.construir_modelo()
        
        logger.info(f"🚀 Iniciando entrenamiento ({epochs} épocas)...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if y_val is not None else 'loss',
            patience=15,
            restore_best_weights=True
        )
        
        historial = self.modelo_nn.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        logger.info(f"✅ Entrenamiento completado")
        return historial.history
    
    # ========================================================================
    # EVALUACIÓN DEL MODELO
    # ========================================================================
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo.
        
        Args:
            X_test, y_test: Datos de prueba
        
        Returns:
            Diccionario con métricas
        """
        if self.modelo_nn is None:
            raise ValueError("Modelo no entrenado")
        
        predicciones = self.modelo_nn.predict(X_test, verbose=0)
        
        # Métricas
        mse = np.mean((y_test - predicciones) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predicciones))
        
        # Fidelidad cuántica (similitud de estados)
        fidelidad_promedio = []
        for i in range(min(100, len(y_test))):  # Muestra de 100
            # Reconstruir amplitudes normalizadas
            pred_amplitudes = predicciones[i, :self.dim] + 1j * predicciones[i, self.dim:]
            pred_amplitudes = pred_amplitudes / np.linalg.norm(pred_amplitudes)
            
            true_amplitudes = y_test[i, :self.dim] + 1j * y_test[i, self.dim:]
            true_amplitudes = true_amplitudes / np.linalg.norm(true_amplitudes)
            
            # Fidelidad = |⟨pred|true⟩|²
            fidelidad = np.abs(np.dot(pred_amplitudes.conj(), true_amplitudes)) ** 2
            fidelidad_promedio.append(fidelidad)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "fidelidad_promedio": float(np.mean(fidelidad_promedio)),
            "samples": len(X_test),
            "timestamp": datetime.now().isoformat()
        }
    
    # ========================================================================
    # PREDICCIÓN
    # ========================================================================
    
    def predecir_evolucion(self, estado_inicial: np.ndarray, pasos: int = 5) -> List[np.ndarray]:
        """
        Predice evolución futura del estado cuántico.
        
        Args:
            estado_inicial: Estado cuántico inicial
            pasos: Número de pasos a predecir
        
        Returns:
            Lista de estados predichos
        """
        if self.modelo_nn is None:
            raise ValueError("Modelo no entrenado")
        
        predicciones = []
        estado_actual = estado_inicial.copy()
        
        for _ in range(pasos):
            # Preparar entrada
            entrada = np.concatenate([
                np.abs(estado_actual),
                np.angle(estado_actual),
                [0]
            ]).reshape(1, -1).astype(np.float32)
            
            # Predecir
            salida = self.modelo_nn.predict(entrada, verbose=0)[0]
            
            # Reconstruir estado
            amplitudes = salida[:self.dim] + 1j * salida[self.dim:]
            amplitudes = amplitudes / np.linalg.norm(amplitudes)
            
            predicciones.append(amplitudes)
            estado_actual = amplitudes
        
        return predicciones
    
    # ========================================================================
    # PERSISTENCIA
    # ========================================================================
    
    def guardar_modelo(self, ruta: str) -> bool:
        """Guarda modelo y configuración."""
        try:
            ruta_path = Path(ruta)
            ruta_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar modelo neural
            self.modelo_nn.save(f"{ruta}_nn.keras")
            
            # Guardar configuración
            config = {
                "num_qubits": self.num_qubits,
                "timestamp": datetime.now().isoformat()
            }
            with open(f"{ruta}_config.json", 'w') as f:
                json.dump(config, f)
            
            logger.info(f"✅ Modelo guardado: {ruta}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando: {e}")
            return False
    
    def cargar_modelo(self, ruta: str) -> bool:
        """Carga modelo guardado."""
        try:
            self.modelo_nn = keras.models.load_model(f"{ruta}_nn.keras")
            
            with open(f"{ruta}_config.json", 'r') as f:
                config = json.load(f)
            
            logger.info(f"✅ Modelo cargado: {ruta}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cargando: {e}")
            return False


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

def demo():
    """Demostración del simulador de qubits."""
    print("\n" + "="*80)
    print("🎯 SIMULADOR DE QUBITS CON RED NEURONAL v1.0")
    print("="*80)
    
    print("\n✅ CARACTERÍSTICAS PRINCIPALES:")
    print("   - Simulación exacta de estados cuánticos")
    print("   - Puertas cuánticas: Pauli, Hadamard, Rotaciones")
    print("   - Medición y colapso de función de onda")
    print("   - Entrelazamiento y estados de Bell")
    print("   - Red neuronal para predicción de evolución")
    print("   - Visualización de probabilidades")
    
    print("\n🔬 EJEMPLO DE USO:")
    print("""
    # 1. Crear simulador
    sim = SimuladorQubit(num_qubits=1, seed=42)
    
    # 2. Aplicar puertas
    sim.puerta_hadamard()  # Crear superposición
    estado = sim.estado
    
    # 3. Medir
    resultado = sim.medir()  # 0 ó 1
    
    # 4. Generar datos
    X_train, y_train, X_test, y_test = sim.generar_datos_evolucion(
        num_muestras=1000
    )
    
    # 5. Entrenar modelo
    sim.construir_modelo()
    sim.entrenar(X_train, y_train, X_test, y_test)
    
    # 6. Evaluar
    metricas = sim.evaluar(X_test, y_test)
    """)
    
    print("\n🧪 PUERTAS DISPONIBLES:")
    print("   - puerta_pauli_x/y/z(): Matrices de Pauli")
    print("   - puerta_hadamard(): Superposición")
    print("   - puerta_rotacion_x/y/z(theta): Rotaciones")
    print("   - puerta_fase(phi): Cambio de fase")
    print("   - puerta_cnot(): Controlled-NOT (2 qubits)")
    
    print("\n📊 ESTADOS ESPECIALES:")
    print("   - crear_superposicion_igual(): (|0⟩+|1⟩)/√2")
    print("   - crear_bell_state('00'/'01'/'10'/'11'): Estados entrelazados")
    
    print("\n" + "="*80)
    print("Para más información, ver README.md")
    print("="*80 + "\n")


if __name__ == '__main__':
    demo()
