# Proyecto 12: Qubits Entrelazados

Simulación de entrelazamiento cuántico, estados de Bell y desigualdades de Bell (CHSH).

## Características

- 4 estados de Bell: $|\Phi^+\rangle$, $|\Phi^-\rangle$, $|\Psi^+\rangle$, $|\Psi^-\rangle$
- Puertas lógicas: CNOT, Hadamard
- Correlaciones cuánticas
- Desigualdad de CHSH
- Medidas correlacionadas

## Estados de Bell

- $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ - Máximo entrelazamiento simétrico
- $|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$ - Antisimétrico
- $|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$ - Simétrico alternado
- $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ - Antisimétrico alternado

## Uso

```bash
python run_entrelazados.py
```

## Análisis

- Correlaciones ZZ por estado
- Violación de desigualdades de Bell
- Estadística de medidas correlacionadas
- Límite clásico vs cuántico
