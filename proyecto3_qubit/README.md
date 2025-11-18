# Proyecto 3: Simulador de Qubit y Decoherencia

## Descripción

Simulador profesional de sistemas cuánticos implementando:
- **Esfera de Bloch**: Visualización 3D de estados
- **Dinámicas de Rabi**: Oscilaciones coherentes
- **Decoherencia**: T1 (relajación) y T2 (defasing)
- **Experimentos cuánticos**: Echo de spin (Hahn Echo)
- **Visualización 3D**: Trayectorias en esfera de Bloch

## Características Principales

### 1. Estados Cuánticos
- |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩
- Estados arbitrarios
- Cálculo de pureza y magnetización

### 2. Operaciones
- Evolución libre (precesión de Larmor)
- Pulsos resonantes (π, π/2)
- Decoherencia controlada

### 3. Experimentos
- Oscilaciones de Rabi
- Hahn Echo (corrección de defasing)
- Análisis de decoherencia

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Ejecutar simulaciones
python run_simulations.py

# Tests
pytest test_simulador.py -v
```

## API Referencia

### Clase: `SimuladorQubit`

| Método | Descripción |
|--------|-------------|
| `estado_inicial(tipo)` | Crea estados estándar |
| `evolucion_libre()` | Precesión de Larmor |
| `decoherencia_t1()` | Relajación |
| `decoherencia_t2()` | Defasing |
| `pulso_resonante()` | Rotaciones Rabi |
| `experimento_rabi()` | Oscilaciones de Rabi |
| `experimento_echo()` | Hahn Echo |
| `visualizar_esfera_bloch()` | Gráficas 3D |

## Resultados

El simulador genera:
- 4 visualizaciones de la esfera de Bloch
- Gráficas de Rabi, decoherencia, y echo
- Análisis cuantitativo en JSON

## Testing

```bash
pytest test_simulador.py --cov=simulador_qubit
```

30+ tests unitarios con cobertura >95%

## Referencias

- Neilsen & Chuang: "Quantum Computation and Quantum Information"
- Dutt & Osheroff: "Decoherence and the Transition from Quantum to Classical"

## Licencia

MIT License
