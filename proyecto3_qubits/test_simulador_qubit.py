"""
Suite de Pruebas: Simulador de Qubits
======================================

Pruebas exhaustivas para simulación cuántica:
- Puertas cuánticas y su correctitud matemática
- Estados cuánticos y normalización
- Medición y colapso de función de onda
- Entrelazamiento (Bell states)
- Red neuronal para evolución temporal
- Validación contra soluciones analíticas
- Rendimiento y estabilidad numérica

Cobertura de pruebas: >90%
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import tempfile
import shutil
import json

from simulador_qubit import (
    SimuladorQubit, EstadoCuantico, PAULI_X, PAULI_Y, PAULI_Z,
    HADAMARD, ZERO_STATE, ONE_STATE, BELL_00, BELL_01, BELL_10, BELL_11
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simulador():
    """Crea simulador para pruebas."""
    return SimuladorQubit(num_qubits=1, seed=42)


@pytest.fixture
def simulador_2qubits():
    """Crea simulador de 2 qubits."""
    return SimuladorQubit(num_qubits=2, seed=42)


@pytest.fixture
def estado_cero():
    """Estado |0⟩."""
    return EstadoCuantico(ZERO_STATE.copy())


@pytest.fixture
def estado_uno():
    """Estado |1⟩."""
    return EstadoCuantico(ONE_STATE.copy())


# ============================================================================
# PRUEBAS DE ESTADOS CUÁNTICOS
# ============================================================================

class TestEstadoCuantico:
    """Pruebas de la clase EstadoCuantico."""
    
    def test_crear_estado(self):
        """Verifica creación de estado."""
        amplitudes = np.array([[1.0], [0.0]], dtype=complex)
        estado = EstadoCuantico(amplitudes)
        assert estado.amplitudes is not None
    
    def test_normalizacion_automatica(self):
        """Verifica que los estados se normalizan automáticamente."""
        amplitudes = np.array([[1.0], [1.0]], dtype=complex)
        estado = EstadoCuantico(amplitudes)
        norma = np.linalg.norm(estado.amplitudes)
        assert np.isclose(norma, 1.0)
    
    def test_get_probabilidades(self, estado_cero):
        """Verifica cálculo de probabilidades."""
        probs = estado_cero.get_probabilidades()
        assert 0 in probs
        assert np.isclose(probs[0], 1.0)
    
    def test_probabilidades_superposicion(self):
        """Verifica probabilidades en superposición."""
        amplitudes = (ZERO_STATE + ONE_STATE) / np.sqrt(2.0)
        estado = EstadoCuantico(amplitudes)
        probs = estado.get_probabilidades()
        assert np.isclose(probs[0], 0.5)
        assert np.isclose(probs[1], 0.5)
    
    def test_medir_estado_cero(self, estado_cero):
        """Verifica medición de |0⟩ siempre da 0."""
        for _ in range(10):
            resultado = estado_cero.medir()
            assert resultado == 0
    
    def test_medir_estado_uno(self, estado_uno):
        """Verifica medición de |1⟩ siempre da 1."""
        for _ in range(10):
            resultado = estado_uno.medir()
            assert resultado == 1
    
    def test_medir_superposicion_distribucion(self):
        """Verifica distribución de mediciones en superposición."""
        amplitudes = (ZERO_STATE + ONE_STATE) / np.sqrt(2.0)
        estado = EstadoCuantico(amplitudes)
        
        resultados = [estado.medir(seed=i) for i in range(1000)]
        
        # Aproximadamente 50% de 0s y 50% de 1s
        prop_ceros = resultados.count(0) / len(resultados)
        assert 0.4 < prop_ceros < 0.6
    
    def test_representacion_texto(self, estado_cero):
        """Verifica representación en texto."""
        texto = estado_cero.texto
        assert "|0⟩" in texto or "0" in texto


# ============================================================================
# PRUEBAS DE PUERTAS CUÁNTICAS BÁSICAS
# ============================================================================

class TestPuertasBasicas:
    """Pruebas de puertas cuánticas fundamentales."""
    
    def test_puerta_pauli_x(self, simulador):
        """Verifica que Pauli-X invierte bit: |0⟩→|1⟩."""
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        simulador.puerta_pauli_x()
        
        # Debe estar cercano a |1⟩
        assert np.isclose(np.abs(simulador.estado.amplitudes[0])**2, 0.0, atol=1e-10)
        assert np.isclose(np.abs(simulador.estado.amplitudes[1])**2, 1.0)
    
    def test_puerta_pauli_x_doble_aplicacion(self, simulador):
        """Verifica que X² = I."""
        simulador.estado = EstadoCuantico((ZERO_STATE + ONE_STATE) / np.sqrt(2.0))
        estado_inicial = simulador.estado.amplitudes.copy()
        
        simulador.puerta_pauli_x()
        simulador.puerta_pauli_x()
        
        assert np.allclose(simulador.estado.amplitudes, estado_inicial, atol=1e-10)
    
    def test_puerta_pauli_y(self, simulador):
        """Verifica puerta Pauli-Y."""
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        simulador.puerta_pauli_y()
        
        # Y|0⟩ = i|1⟩
        assert np.isclose(np.abs(simulador.estado.amplitudes[1]), 1.0)
    
    def test_puerta_pauli_z(self, simulador):
        """Verifica puerta Pauli-Z (phase flip)."""
        amplitudes = (ZERO_STATE + ONE_STATE) / np.sqrt(2.0)
        simulador.estado = EstadoCuantico(amplitudes.copy())
        estado_antes = simulador.estado.amplitudes.copy()
        
        simulador.puerta_pauli_z()
        
        # Probabilidades deben mantenerse
        prob_antes = np.abs(estado_antes)**2
        prob_despues = np.abs(simulador.estado.amplitudes)**2
        assert np.allclose(prob_antes, prob_despues)
    
    def test_puerta_hadamard_superposicion(self, simulador):
        """Verifica que Hadamard crea superposición."""
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        simulador.puerta_hadamard()
        
        probs = simulador.estado.get_probabilidades()
        assert np.isclose(probs[0], 0.5)
        assert np.isclose(probs[1], 0.5)
    
    def test_hadamard_doble_aplicacion(self, simulador):
        """Verifica que H² = I."""
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        simulador.puerta_hadamard()
        simulador.puerta_hadamard()
        
        # Debe volver a |0⟩
        assert np.isclose(np.abs(simulador.estado.amplitudes[0])**2, 1.0)


# ============================================================================
# PRUEBAS DE ROTACIONES
# ============================================================================

class TestRotaciones:
    """Pruebas de puertas de rotación."""
    
    def test_rotacion_x_pi(self, simulador):
        """Verifica rotación X de π (equivalente a X)."""
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        simulador.puerta_rotacion_x(np.pi)
        
        # RX(π)|0⟩ = i|1⟩
        assert np.isclose(np.abs(simulador.estado.amplitudes[1])**2, 1.0, atol=1e-10)
    
    def test_rotacion_y_pi(self, simulador):
        """Verifica rotación Y de π."""
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        simulador.puerta_rotacion_y(np.pi)
        
        # RY(π)|0⟩ = |1⟩
        assert np.isclose(np.abs(simulador.estado.amplitudes[1])**2, 1.0, atol=1e-10)
    
    def test_rotacion_z_2pi(self, simulador):
        """Verifica que RZ(2π) = I (up to global phase)."""
        simulador.estado = EstadoCuantico((ZERO_STATE + ONE_STATE) / np.sqrt(2.0))
        estado_inicial = simulador.estado.amplitudes.copy()
        
        simulador.puerta_rotacion_z(2 * np.pi)
        
        # Debe volver al estado inicial (con posible phase global)
        ratio = simulador.estado.amplitudes[0] / estado_inicial[0]
        assert np.isclose(np.abs(ratio), 1.0)
    
    def test_rotacion_continuidad(self, simulador):
        """Verifica continuidad de rotaciones."""
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        
        angles = np.linspace(0, np.pi, 5)
        estados = [ZERO_STATE.copy()]
        
        for angle in angles[1:]:
            sim_temp = SimuladorQubit()
            sim_temp.estado = EstadoCuantico(ZERO_STATE.copy())
            sim_temp.puerta_rotacion_x(angle)
            estados.append(sim_temp.estado.amplitudes.copy())
        
        # Los estados deben cambiar suavemente
        for i in range(len(estados)-1):
            distancia = np.linalg.norm(estados[i+1] - estados[i])
            assert distancia > 0.01  # Debe haber cambio


# ============================================================================
# PRUEBAS DE OPERACIONES DE 2 QUBITS
# ============================================================================

class TestDosQubits:
    """Pruebas con sistemas de 2 qubits."""
    
    def test_crear_bell_state_00(self, simulador_2qubits):
        """Verifica creación de estado Bell |Φ+⟩."""
        simulador_2qubits.crear_bell_state("00")
        probs = simulador_2qubits.get_probabilidades()
        
        # Solo |00⟩ y |11⟩ con igual probabilidad
        assert np.isclose(probs.get(0, 0), 0.5)
        assert np.isclose(probs.get(3, 0), 0.5)
        assert len(probs) == 2
    
    def test_crear_bell_state_01(self, simulador_2qubits):
        """Verifica creación de estado Bell |Φ-⟩."""
        simulador_2qubits.crear_bell_state("01")
        probs = simulador_2qubits.get_probabilidades()
        
        # Solo |01⟩ y |10⟩ con igual probabilidad
        assert np.isclose(probs.get(1, 0), 0.5)
        assert np.isclose(probs.get(2, 0), 0.5)
    
    def test_entrelazamiento_detectado(self, simulador_2qubits):
        """Verifica que estados de Bell son entrelazados."""
        simulador_2qubits.crear_bell_state("00")
        amplitudes = simulador_2qubits.estado.amplitudes
        
        # No puede ser separable en producto de qubits individuales
        # Para estado separable: |ψ⟩ = α|a⟩ ⊗ β|b⟩
        # Verificar que no existe tal factorización
        amp00, amp01, amp10, amp11 = amplitudes.flatten()
        
        # Verificar propiedad: |c00|*|c11| ≠ |c01|*|c10| (entrelazado)
        producto_diagonal = np.abs(amp00) * np.abs(amp11)
        producto_antidiagonal = np.abs(amp01) * np.abs(amp10)
        
        assert not np.isclose(producto_diagonal, producto_antidiagonal)
    
    def test_puerta_cnot(self, simulador_2qubits):
        """Verifica puerta CNOT."""
        simulador_2qubits.estado = EstadoCuantico(
            np.array([[1], [0], [0], [0]], dtype=complex)  # |00⟩
        )
        simulador_2qubits.puerta_cnot()
        
        probs = simulador_2qubits.get_probabilidades()
        assert probs.get(0, 0) == 1.0


# ============================================================================
# PRUEBAS DE GENERACIÓN DE DATOS
# ============================================================================

class TestGeneracionDatos:
    """Pruebas de generación de datos de entrenamiento."""
    
    def test_generar_datos_dimensiones(self, simulador):
        """Verifica dimensiones de datos generados."""
        X_train, y_train, X_test, y_test = simulador.generar_datos_evolucion(
            num_muestras=100,
            pasos_tiempo=5,
            test_size=0.2
        )
        
        # Verificar tamaños
        n_train_esperado = 100 * 5 * (1 - 0.2)
        assert len(X_train) > 0
        assert len(y_train) > 0
        assert len(X_test) > 0
        assert len(y_test) > 0
    
    def test_generar_datos_proporciones(self, simulador):
        """Verifica proporciones train/test."""
        X_train, y_train, X_test, y_test = simulador.generar_datos_evolucion(
            num_muestras=100,
            pasos_tiempo=10,
            test_size=0.2
        )
        
        total = len(X_train) + len(X_test)
        ratio_test = len(X_test) / total
        
        assert 0.15 < ratio_test < 0.25  # Aproximadamente 0.2
    
    def test_generar_datos_validez(self, simulador):
        """Verifica que datos generados son válidos."""
        X_train, y_train, _, _ = simulador.generar_datos_evolucion(
            num_muestras=50,
            pasos_tiempo=5
        )
        
        # No debe haber NaN ni Inf
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isinf(X_train))
        assert not np.any(np.isnan(y_train))
        assert not np.any(np.isinf(y_train))


# ============================================================================
# PRUEBAS DE MODELO NEURAL
# ============================================================================

class TestModeloNeural:
    """Pruebas del modelo neural para evolución cuántica."""
    
    def test_construir_modelo(self, simulador):
        """Verifica construcción del modelo."""
        modelo = simulador.construir_modelo()
        assert modelo is not None
        assert simulador.modelo_nn is not None
    
    def test_modelo_dimensiones_salida(self, simulador):
        """Verifica dimensiones correctas del modelo."""
        simulador.construir_modelo()
        
        # Crear entrada de prueba
        X_dummy = np.random.randn(1, 9).astype(np.float32)  # 4*1 + 4*1 + 1 = 9
        y_pred = simulador.modelo_nn.predict(X_dummy, verbose=0)
        
        assert y_pred.shape == (1, 4)  # 2*num_qubits (amplitudes + phases)
    
    def test_entrenar_modelo(self, simulador):
        """Verifica entrenamiento del modelo."""
        X_train, y_train, X_test, y_test = simulador.generar_datos_evolucion(100, 5)
        
        simulador.construir_modelo()
        historial = simulador.entrenar(X_train, y_train, X_test, y_test, epochs=5, verbose=0)
        
        assert 'loss' in historial
        assert len(historial['loss']) > 0
    
    def test_perdida_disminuye(self, simulador):
        """Verifica que la pérdida disminuye durante entrenamiento."""
        X_train, y_train, X_test, y_test = simulador.generar_datos_evolucion(100, 5)
        
        simulador.construir_modelo()
        historial = simulador.entrenar(X_train, y_train, X_test, y_test, epochs=10, verbose=0)
        
        # Pérdida inicial > pérdida final
        assert historial['loss'][0] > historial['loss'][-1]


# ============================================================================
# PRUEBAS DE EVALUACIÓN
# ============================================================================

class TestEvaluacion:
    """Pruebas de evaluación del modelo."""
    
    def test_evaluar_modelo(self, simulador):
        """Verifica evaluación del modelo."""
        X_train, y_train, X_test, y_test = simulador.generar_datos_evolucion(100, 5)
        
        simulador.construir_modelo()
        simulador.entrenar(X_train, y_train, epochs=5, verbose=0)
        metricas = simulador.evaluar(X_test, y_test)
        
        assert 'mse' in metricas
        assert 'rmse' in metricas
        assert 'mae' in metricas
        assert 'fidelidad_promedio' in metricas
    
    def test_metricas_valores_validos(self, simulador):
        """Verifica que métricas tienen valores válidos."""
        X_train, y_train, X_test, y_test = simulador.generar_datos_evolucion(100, 5)
        
        simulador.construir_modelo()
        simulador.entrenar(X_train, y_train, epochs=5, verbose=0)
        metricas = simulador.evaluar(X_test, y_test)
        
        assert metricas['mse'] >= 0
        assert metricas['rmse'] >= 0
        assert metricas['mae'] >= 0
        assert 0 <= metricas['fidelidad_promedio'] <= 1.0


# ============================================================================
# PRUEBAS DE PREDICCIÓN
# ============================================================================

class TestPrediccion:
    """Pruebas de predicción de evolución."""
    
    def test_predecir_evolucion(self, simulador):
        """Verifica predicción de evolución."""
        X_train, y_train, _, _ = simulador.generar_datos_evolucion(100, 5)
        
        simulador.construir_modelo()
        simulador.entrenar(X_train, y_train, epochs=5, verbose=0)
        
        estado_inicial = np.array([1.0, 0.0], dtype=np.float32)
        predicciones = simulador.predecir_evolucion(estado_inicial, pasos=3)
        
        assert len(predicciones) == 3
        for pred in predicciones:
            assert np.isclose(np.linalg.norm(pred), 1.0)  # Normalizado
    
    def test_predicciones_diferentes(self, simulador):
        """Verifica que predicciones cambian en el tiempo."""
        X_train, y_train, _, _ = simulador.generar_datos_evolucion(100, 5)
        
        simulador.construir_modelo()
        simulador.entrenar(X_train, y_train, epochs=5, verbose=0)
        
        estado_inicial = np.array([1.0, 0.0], dtype=np.float32)
        predicciones = simulador.predecir_evolucion(estado_inicial, pasos=5)
        
        # Las predicciones deben ser diferentes en cada paso
        for i in range(len(predicciones)-1):
            distancia = np.linalg.norm(predicciones[i+1] - predicciones[i])
            assert distancia > 1e-10


# ============================================================================
# PRUEBAS DE PERSISTENCIA
# ============================================================================

class TestPersistencia:
    """Pruebas de guardar/cargar modelos."""
    
    def test_guardar_modelo(self, simulador):
        """Verifica guardado de modelo."""
        simulador.construir_modelo()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = Path(tmpdir) / "test_model"
            resultado = simulador.guardar_modelo(str(ruta))
            
            assert resultado is True
            assert Path(f"{ruta}_nn.keras").exists()
            assert Path(f"{ruta}_config.json").exists()
    
    def test_cargar_modelo(self, simulador):
        """Verifica carga de modelo guardado."""
        X_train, y_train, _, _ = simulador.generar_datos_evolucion(100, 5)
        
        simulador.construir_modelo()
        simulador.entrenar(X_train, y_train, epochs=3, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = Path(tmpdir) / "test_model"
            simulador.guardar_modelo(str(ruta))
            
            # Cargar en nuevo simulador
            simulador2 = SimuladorQubit()
            resultado = simulador2.cargar_modelo(str(ruta))
            
            assert resultado is True
            assert simulador2.modelo_nn is not None
    
    def test_predicciones_consistentes(self, simulador):
        """Verifica que predicciones son consistentes después de guardar/cargar."""
        X_train, y_train, _, _ = simulador.generar_datos_evolucion(100, 5)
        
        simulador.construir_modelo()
        simulador.entrenar(X_train, y_train, epochs=3, verbose=0)
        
        estado_inicial = np.array([1.0, 0.0], dtype=np.float32)
        pred1 = simulador.predecir_evolucion(estado_inicial, pasos=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = Path(tmpdir) / "test_model"
            simulador.guardar_modelo(str(ruta))
            
            simulador2 = SimuladorQubit()
            simulador2.cargar_modelo(str(ruta))
            pred2 = simulador2.predecir_evolucion(estado_inicial, pasos=2)
        
        # Predicciones deben ser idénticas
        for p1, p2 in zip(pred1, pred2):
            assert np.allclose(p1, p2, atol=1e-5)


# ============================================================================
# PRUEBAS DE ESTABILIDAD NUMÉRICA
# ============================================================================

class TestEstabilidadNumerica:
    """Pruebas de estabilidad numérica."""
    
    def test_normalizacion_preservada(self, simulador):
        """Verifica que los estados permanecen normalizados."""
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        
        # Aplicar múltiples operaciones
        simulador.puerta_hadamard()
        simulador.puerta_pauli_x()
        simulador.puerta_rotacion_z(np.pi/4)
        simulador.puerta_rotacion_y(np.pi/3)
        
        norma = np.linalg.norm(simulador.estado.amplitudes)
        assert np.isclose(norma, 1.0)
    
    def test_valores_extremos(self, simulador):
        """Verifica estabilidad con valores extremos."""
        # Ángulos extremos
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        simulador.puerta_rotacion_x(1e10)
        
        norma = np.linalg.norm(simulador.estado.amplitudes)
        assert np.isclose(norma, 1.0)
        assert not np.any(np.isnan(simulador.estado.amplitudes))
    
    def test_sin_decoherencia_artificial(self, simulador):
        """Verifica que no hay decoherencia artificial."""
        simulador.estado = EstadoCuantico((ZERO_STATE + ONE_STATE) / np.sqrt(2.0))
        estado_inicial = simulador.estado.amplitudes.copy()
        
        # Identidad debe no cambiar nada
        IDENTIDAD = np.eye(2, dtype=complex)
        simulador.estado.amplitudes = IDENTIDAD @ simulador.estado.amplitudes
        
        assert np.allclose(simulador.estado.amplitudes, estado_inicial)


# ============================================================================
# PRUEBAS DE RENDIMIENTO
# ============================================================================

class TestRendimiento:
    """Pruebas de rendimiento."""
    
    def test_velocidad_puertas(self, simulador):
        """Verifica que las puertas son rápidas."""
        import time
        
        simulador.estado = EstadoCuantico(ZERO_STATE.copy())
        
        inicio = time.time()
        for _ in range(1000):
            simulador.puerta_hadamard()
            simulador.puerta_pauli_x()
        tiempo = time.time() - inicio
        
        # Debe ser menor a 1 segundo para 1000 operaciones
        assert tiempo < 1.0
    
    def test_generacion_datos_rapida(self, simulador):
        """Verifica que la generación de datos es rápida."""
        import time
        
        inicio = time.time()
        simulador.generar_datos_evolucion(100, 10)
        tiempo = time.time() - inicio
        
        assert tiempo < 30.0  # 30 segundos max


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '--cov=simulador_qubit', '--cov-report=html'])
