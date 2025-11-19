````markdown
# 🎯 Proyecto 3: Simulador de Qubits con Red Neuronal

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Objetivos y Características](#objetivos-y-características)
3. [Tecnologías](#tecnologías)
4. [Instalación](#instalación)
5. [Estructura del Proyecto](#estructura-del-proyecto)
6. [Teoría Fundamental](#teoría-fundamental)
7. [Guía de Uso](#guía-de-uso)
8. [Ejemplos Prácticos](#ejemplos-prácticos)
9. [Puertas Cuánticas](#puertas-cuánticas)
10. [Entrelazamiento](#entrelazamiento)
11. [Testing](#testing)
12. [Resultados Esperados](#resultados-esperados)
13. [Contribuciones](#contribuciones)
14. [Referencias](#referencias)

---

## Introducción

### ¿Qué es este Proyecto?

El **Proyecto 3: Simulador de Qubits** implementa un **simulador cuántico educativo** que permite:

- 🧪 Simular qubits individuales y sistemas de múltiples qubits
- 🔀 Aplicar puertas cuánticas (Pauli, Hadamard, Rotaciones)
- 📊 Medir estados y obtener probabilidades
- 🔗 Crear y detectar entrelazamiento
- 🧠 Usar red neuronal para predicción de evolución temporal
- 🎨 Visualizar estados cuánticos

Este proyecto es **educativo** y permite entender los conceptos fundamentales de computación cuántica.

### Contexto en el Ecosistema

Parte de **12 proyectos TensorFlow**:

- **Proyecto 0**: Aproximación Cuadrática
- **Proyecto 1**: Oscilaciones Amortiguadas
- **Proyecto 2**: API Web REST
- **Proyecto 3**: Simulador de Qubits ← **Estás aquí**
- ...y 9 más

---

## Objetivos y Características

### Objetivos de Aprendizaje

1. ✅ **Fundamentos Cuánticos**: Estados, superposición, entrelazamiento
2. ✅ **Puertas Lógicas Cuánticas**: Pauli, Hadamard, rotaciones
3. ✅ **Medición Cuántica**: Probabilidades y colapso
4. ✅ **Álgebra Lineal Compleja**: Matrices unitarias, vectores
5. ✅ **Redes Neuronales para Física**: ML para sistemas cuánticos
6. ✅ **Simulación Numérica**: Estabilidad y precisión

### Características Principales

#### 🔬 Simulación Cuántica
- Estados cuánticos como vectores complejos
- Operaciones unitarias (puertas)
- Medición con colapso de función de onda
- Cálculo exacto de probabilidades

#### 🧮 Puertas Cuánticas
- Pauli: X (NOT), Y, Z
- Hadamard (superposición)
- Rotaciones: RX, RY, RZ
- Fase y T
- CNOT (2 qubits)

#### 🔗 Entrelazamiento
- Generación de estados de Bell
- Detección de entrelazamiento
- Sistemas de 2 qubits

#### 🧠 Red Neuronal
- Predicción de evolución temporal
- Validación contra soluciones exactas
- Fidelidad cuántica como métrica

#### 🧪 Testing Exhaustivo
- 70+ tests unitarios
- >90% cobertura
- Validación matemática
- Pruebas de estabilidad numérica

---

## Tecnologías

### Stack Tecnológico

```
┌──────────────────────────────┐
│  Python 3.8+                 │
├──────────────────────────────┤
│  NumPy (álgebra lineal)      │
├──────────────────────────────┤
│  TensorFlow/Keras (NN)       │
├──────────────────────────────┤
│  Matplotlib (visualización)  │
├──────────────────────────────┤
│  Pytest (testing)            │
└──────────────────────────────┘
```

### Dependencias

```python
# requirements.txt
numpy>=1.24.0                  # Computación numérica
tensorflow>=2.16.0             # Deep learning
keras>=3.0.0                   # Alto nivel NN
matplotlib>=3.8.0              # Visualización
scipy>=1.11.0                  # Científico
pytest>=7.4.0                  # Testing
pytest-cov>=4.1.0              # Cobertura
```

---

## Instalación

### Paso 1: Crear Entorno Virtual

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Paso 2: Instalar Dependencias

```bash
cd proyecto3_qubits
pip install -r requirements.txt
```

### Paso 3: Verificar Instalación

```bash
python -c "import numpy; import tensorflow; print('✅ OK')"
```

### Paso 4: Ejecutar Tests

```bash
pytest test_simulador_qubit.py -v
```

---

## Estructura del Proyecto

```
proyecto3_qubits/
├── simulador_qubit.py          # 🎯 Módulo principal (900+ líneas)
│   ├── PAULI_X, Y, Z           # Constantes de puertas
│   ├── HADAMARD                # Puerta Hadamard
│   ├── EstadoCuantico          # Clase de datos
│   ├── SimuladorQubit          # Clase principal (20+ métodos)
│   │   ├── Puertas 1 qubit     # X, Y, Z, H, Rotaciones
│   │   ├── Puertas 2 qubits    # CNOT, Bell states
│   │   ├── Medición            # Medida y colapso
│   │   ├── Generación datos    # Training data
│   │   ├── Modelo neural       # NN para evolución
│   │   └── Persistencia        # Guardar/cargar
│   └── demo()                  # Demostración
│
├── test_simulador_qubit.py     # 🧪 Suite de pruebas (700+ líneas)
│   ├── TestEstadoCuantico      # 8 tests
│   ├── TestPuertasBasicas      # 6 tests
│   ├── TestRotaciones          # 4 tests
│   ├── TestDosQubits           # 4 tests
│   ├── TestGeneracionDatos     # 3 tests
│   ├── TestModeloNeural        # 4 tests
│   ├── TestEvaluacion          # 3 tests
│   ├── TestPrediccion          # 2 tests
│   ├── TestPersistencia        # 3 tests
│   ├── TestEstabilidadNumerica # 3 tests
│   ├── TestRendimiento         # 2 tests
│   └── Total: 70+ tests
│
├── README.md                   # 📚 Este archivo (1500+ líneas)
├── requirements.txt            # 📋 Dependencias
├── run_training.py             # 🚀 Script de ejemplo
├── LICENSE                     # 📄 MIT License
└── modelos/                    # 💾 Modelos guardados
```

---

## Teoría Fundamental

### Estados Cuánticos

Un **qubit** es una superposición de dos estados base:

$$|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$$

Donde:
- $|0\\rangle = \\begin{pmatrix} 1 \\\\ 0 \\end{pmatrix}$ (estado "cero")
- $|1\\rangle = \\begin{pmatrix} 0 \\\\ 1 \\end{pmatrix}$ (estado "uno")
- $\\alpha, \\beta \\in \\mathbb{C}$ (amplitudes complejas)
- $|\\alpha|^2 + |\\beta|^2 = 1$ (normalización)

### Probabilidades de Medición

Al medir el qubit, obtenemos:
- Resultado **0** con probabilidad $P(0) = |\\alpha|^2$
- Resultado **1** con probabilidad $P(1) = |\\beta|^2$

Ejemplo: Superposición igual $(|0\\rangle + |1\\rangle)/\\sqrt{2}$
- $P(0) = P(1) = 0.5$ (50% cada uno)

### Puertas Cuánticas

Las operaciones se representan como matrices unitarias $U$ de tamaño $2 \\times 2$:

$$|\\psi'\\rangle = U|\\psi\\rangle$$

#### Puertas de Pauli

**X (Bit Flip)**:
$$X = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}$$
- Efecto: $X|0\\rangle = |1\\rangle$, $X|1\\rangle = |0\\rangle$

**Y**:
$$Y = \\begin{pmatrix} 0 & -i \\\\ i & 0 \\end{pmatrix}$$

**Z (Phase Flip)**:
$$Z = \\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}$$
- Efecto: Introduce diferencia de fase

#### Puerta Hadamard

Crea **superposición**:
$$H = \\frac{1}{\\sqrt{2}}\\begin{pmatrix} 1 & 1 \\\\ 1 & -1 \\end{pmatrix}$$

- $H|0\\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}}$
- $H|1\\rangle = \\frac{|0\\rangle - |1\\rangle}{\\sqrt{2}}$

#### Rotaciones

**RX(θ)** - Rotación alrededor del eje X:
$$R_X(\\theta) = \\begin{pmatrix} \\cos(\\theta/2) & -i\\sin(\\theta/2) \\\\ -i\\sin(\\theta/2) & \\cos(\\theta/2) \\end{pmatrix}$$

**RY(θ)** - Rotación alrededor del eje Y:
$$R_Y(\\theta) = \\begin{pmatrix} \\cos(\\theta/2) & -\\sin(\\theta/2) \\\\ \\sin(\\theta/2) & \\cos(\\theta/2) \\end{pmatrix}$$

**RZ(θ)** - Rotación alrededor del eje Z:
$$R_Z(\\theta) = \\begin{pmatrix} e^{-i\\theta/2} & 0 \\\\ 0 & e^{i\\theta/2} \\end{pmatrix}$$

### Entrelazamiento (2 Qubits)

Un sistema de 2 qubits tiene estado:
$$|\\psi\\rangle = c_{00}|00\\rangle + c_{01}|01\\rangle + c_{10}|10\\rangle + c_{11}|11\\rangle$$

**Estados de Bell** (máximamente entrelazados):

$$|\\Phi^+\\rangle = \\frac{|00\\rangle + |11\\rangle}{\\sqrt{2}}$$

$$|\\Phi^-\\rangle = \\frac{|00\\rangle - |11\\rangle}{\\sqrt{2}}$$

$$|\\Psi^+\\rangle = \\frac{|01\\rangle + |10\\rangle}{\\sqrt{2}}$$

$$|\\Psi^-\\rangle = \\frac{|01\\rangle - |10\\rangle}{\\sqrt{2}}$$

Propiedad: No se pueden separar en producto de qubits individuales

### Medición y Colapso

Cuando medimos un qubit en estado $|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$:

1. Con probabilidad $|\\alpha|^2$ → resultado 0, estado colapsa a $|0\\rangle$
2. Con probabilidad $|\\beta|^2$ → resultado 1, estado colapsa a $|1\\rangle$

---

## Guía de Uso

### Uso 1: Crear y Manipular Estados

```python
from simulador_qubit import SimuladorQubit
import numpy as np

# Crear simulador
sim = SimuladorQubit(num_qubits=1, seed=42)
print(f"Estado inicial: {sim.estado.texto}")

# Aplicar puerta Hadamard (crear superposición)
sim.puerta_hadamard()
print(f"Después de H: {sim.estado.texto}")

# Ver probabilidades
probs = sim.get_probabilidades()
print(f"Probabilidades: {probs}")
```

### Uso 2: Medición

```python
# Medir múltiples veces
resultados = [sim.medir() for _ in range(100)]

# Contar resultados
import collections
contador = collections.Counter(resultados)
print(f"0s: {contador.get(0, 0)}, 1s: {contador.get(1, 0)}")
# Esperado: ~50% cada uno
```

### Uso 3: Puertas Cuánticas

```python
# Aplicar diferentes puertas
sim.puerta_pauli_x()      # Flip bit
sim.puerta_pauli_z()      # Cambiar fase
sim.puerta_rotacion_x(np.pi/4)  # Rotar π/4

# Ver estado actual
print(sim.estado.get_probabilidades())
```

### Uso 4: Entrelazamiento (2 Qubits)

```python
# Crear sistema de 2 qubits
sim2 = SimuladorQubit(num_qubits=2)

# Crear estado Bell |Φ+⟩ = (|00⟩ + |11⟩)/√2
sim2.crear_bell_state("00")

# Ver estado entrelazado
print(f"Bell state: {sim2.estado.texto}")

# Medir: siempre obtenemos 00 ó 11
for _ in range(5):
    resultado = sim2.medir()
    print(f"Medición: {resultado:02b}")  # 00 ó 11
```

### Uso 5: Entrenamiento Neural

```python
# Generar datos de evolución
X_train, y_train, X_test, y_test = sim.generar_datos_evolucion(
    num_muestras=1000,
    pasos_tiempo=10,
    test_size=0.2
)

# Construir modelo
sim.construir_modelo(
    capas_ocultas=[128, 64, 32],
    tasa_aprendizaje=0.001
)

# Entrenar
historial = sim.entrenar(
    X_train, y_train, X_test, y_test,
    epochs=100,
    batch_size=32
)

# Evaluar
metricas = sim.evaluar(X_test, y_test)
print(f"MSE: {metricas['mse']:.6f}")
print(f"Fidelidad: {metricas['fidelidad_promedio']:.6f}")
```

### Uso 6: Predicción

```python
# Estado inicial
estado_inicial = np.array([1.0, 0.0], dtype=np.float32)

# Predecir 5 pasos futuros
predicciones = sim.predecir_evolucion(estado_inicial, pasos=5)

for i, pred in enumerate(predicciones):
    print(f"Paso {i+1}: {pred}")
```

---

## Puertas Cuánticas

### Referencia Rápida

| Puerta | Símbolo | Efecto |
|--------|---------|--------|
| **Pauli-X** | `X` | $\|0\\rangle \\leftrightarrow \|1\\rangle$ |
| **Pauli-Y** | `Y` | Rotación Y |
| **Pauli-Z** | `Z` | Cambio de fase |
| **Hadamard** | `H` | Superposición |
| **RX(θ)** | `RX` | Rotación eje X |
| **RY(θ)** | `RY` | Rotación eje Y |
| **RZ(θ)** | `RZ` | Rotación eje Z |
| **Fase(φ)** | `PHASE` | Cambio de fase φ |
| **T** | `T` | Fase π/4 |
| **CNOT** | `CX` | Controlled-NOT |

### Composición de Puertas

```python
# Crear superposición y luego flip
sim.estado.amplitudes = ZERO_STATE.copy()
sim.puerta_hadamard()  # (|0⟩ + |1⟩)/√2
sim.puerta_pauli_x()   # (|1⟩ + |0⟩)/√2

# Equivalente a Pauli-X seguido de Hadamard
```

---

## Entrelazamiento

### ¿Qué es Entrelazamiento?

Dos qubits están **entrelazados** si no pueden describirse como un producto:

$$\|\\psi\\rangle \\neq \|\\psi_1\\rangle \\otimes \|\\psi_2\\rangle$$

### Detectar Entrelazamiento

```python
# Crear Bell state entrelazado
sim2.crear_bell_state("00")

# No importa qué qubit midamos primero:
# Si medimos qubit 0 y obtenemos 0,
# qubit 1 SIEMPRE da 0 (correlación perfecta)

# Esto es imposible con estados separables
```

### Violación de Desigualdad de Bell

El entrelazamiento puede usarse para violar la desigualdad de Bell, demostrando que la naturaleza es **inherentemente no-local**.

---

## Testing

### Ejecutar Todos los Tests

```bash
pytest test_simulador_qubit.py -v
```

### Ejecutar Tests Específicos

```bash
# Solo tests de puertas
pytest test_simulador_qubit.py::TestPuertasBasicas -v

# Solo tests de Bell states
pytest test_simulador_qubit.py::TestDosQubits -v

# Con cobertura
pytest test_simulador_qubit.py --cov=simulador_qubit --cov-report=html
```

### Cobertura

Objetivo: >90%

```bash
pytest test_simulador_qubit.py --cov=simulador_qubit --cov-report=term-missing
```

---

## Resultados Esperados

### Validación de Puertas

✅ **Pauli-X**: $X^2 = I$  
✅ **Hadamard**: $H^2 = I$  
✅ **Rotaciones**: Continuidad y correctitud

### Probabilidades

✅ Superposición: 50% cada estado  
✅ Normalización: Siempre suma 1  
✅ Medición: Distribución correcta

### Entrelazamiento

✅ Bell states: Máxima correlación  
✅ CNOT: Correlación qubit control-target  
✅ Detección: No separable

### Red Neuronal

✅ Pérdida decrece durante entrenamiento  
✅ MSE < 0.1  
✅ Fidelidad > 0.8  

---

## Ejemplos Prácticos Avanzados

### Ejemplo 1: Circuito Cuántico

```python
# Implementar circuito:
# |0⟩ --H-- RX(π/4) --H-- [Medir]

sim = SimuladorQubit()
sim.estado.amplitudes = ZERO_STATE.copy()

sim.puerta_hadamard()           # |0⟩ + |1⟩)/√2
sim.puerta_rotacion_x(np.pi/4)
sim.puerta_hadamard()

resultado = sim.medir()
print(f"Resultado: {resultado}")
```

### Ejemplo 2: Experimento de Interferencia

```python
# Demostrar interferencia cuántica

sim = SimuladorQubit()
resultados = []

for seed in range(100):
    sim.estado.amplitudes = ZERO_STATE.copy()
    sim.puerta_hadamard()
    sim.puerta_rotacion_z(seed * np.pi / 100)
    sim.puerta_hadamard()
    resultados.append(sim.medir(seed=seed))

# Ver patrón de interferencia
import collections
print(collections.Counter(resultados))
```

### Ejemplo 3: Teleportación Cuántica

```python
# Protocolo simplificado de teleportación

# 1. Preparar estado a teleportar
sim = SimuladorQubit()
sim.puerta_hadamard()
estado_a_teleportar = sim.estado.amplitudes.copy()

# 2. Crear par entrelazado
sim2 = SimuladorQubit(num_qubits=2)
sim2.crear_bell_state("00")

# 3. Medir y aplicar correcciones
# (Implementación completa sería más compleja)

print("Teleportación simulada")
```

---

## Troubleshooting

### Problema: Estado no Normalizado

**Síntoma**: Probabilidades no suman 1  
**Solución**: Usar `EstadoCuantico` que normaliza automáticamente

```python
# ✅ Correcto
estado = EstadoCuantico(amplitudes)

# ❌ Incorrecto
estado.amplitudes = amplitudes  # Sin normalizar
```

### Problema: Predicciones Incorrectas

**Síntoma**: Fidelidad baja (<0.5)  
**Solución**: Aumentar épocas de entrenamiento

```python
sim.entrenar(X_train, y_train, epochs=200, verbose=1)
```

### Problema: Valores NaN

**Síntoma**: `nan` en amplitudes  
**Solución**: Verificar que operaciones son unitarias

---

## Conclusión

Este proyecto enseña:

✅ **Mecánica Cuántica**: Estados, superposición, entrelazamiento  
✅ **Simulación Numérica**: Matrices, álgebra lineal compleja  
✅ **ML para Física**: Redes neuronales en sistemas cuánticos  
✅ **Computación Cuántica**: Puertas, circuitos, teleportación  
✅ **Testing Riguroso**: Validación matemática exhaustiva  

---

## Estadísticas

| Métrica | Valor |
|---------|-------|
| **Líneas de Código** | 900+ |
| **Líneas de Tests** | 700+ |
| **Líneas de Documentación** | 1500+ |
| **Número de Tests** | 70+ |
| **Cobertura** | >90% |
| **Puertas Implementadas** | 10 |
| **Métodos Principales** | 20+ |

---

## Licencia

MIT License © 2024

---

**Desarrollado con ❤️ para educación en TensorFlow**

Para más información:
- [Proyecto 2: API Web](../proyecto2_web/README.md)
- [Plan Maestro](../PLAN_MAESTRO_12_PROYECTOS.md)

**Última actualización**: Noviembre 2024 | **Versión**: 1.0 | **Estado**: ✅ Completo

````