"""Script ejecución Proyecto 12 - Qubits Entrelazados."""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from simulador_qubits_entrelazados import SimuladorDosQubits


def main():
    print("=" * 70)
    print("QUBITS ENTRELAZADOS Y ESTADOS DE BELL")
    print("=" * 70)
    
    sim = SimuladorDosQubits()
    
    # Parte 1: Estados de Bell
    print("\n1. Creando estados de Bell...")
    estados_bell = sim.crear_estados_bell()
    
    for nombre, estado in estados_bell.items():
        print(f"   |{nombre}>: {estado.full().flatten()[:4]}")
    
    # Parte 2: Generar par entrelazado
    print("\n2. Generando par entrelazado |Φ+>...")
    estado_phi_plus = sim.generar_par_entrelazado()
    print(f"   Estado: {estado_phi_plus.full().flatten()}")
    
    # Parte 3: Correlaciones
    print("\n3. Calculando correlaciones ZZ...")
    for nombre, estado in estados_bell.items():
        corr = sim.calcular_correlacion(estado)
        print(f"   |{nombre}>: <Z1 Z2> = {corr:.6f}")
    
    # Parte 4: Desigualdad CHSH
    print("\n4. Desigualdad de CHSH (Bell)...")
    for nombre, estado in estados_bell.items():
        S = sim.calcular_desigualdad_bell(estado)
        print(f"   |{nombre}>: S = {S:.6f} (clásico: ≤2, cuántico: ≤2√2≈2.828)")
    
    # Parte 5: Medidas simuladas
    print("\n5. Simulando medidas con colapso...")
    num_medidas = 100
    resultados_q1 = []
    resultados_q2 = []
    
    estado_medida = estado_phi_plus.copy()
    for _ in range(num_medidas):
        res1, estado_medida = sim.medir_qubit(estado_medida, 0)
        res2, estado_medida = sim.medir_qubit(estado_medida, 1)
        resultados_q1.append(res1)
        resultados_q2.append(res2)
    
    correlacion_medida = np.mean(np.array(resultados_q1) == np.array(resultados_q2))
    print(f"   Correlación observada: {correlacion_medida:.4f}")
    print(f"   Primeras 10 medidas Q1: {resultados_q1[:10]}")
    print(f"   Primeras 10 medidas Q2: {resultados_q2[:10]}")
    
    # Gráficos
    print("\n6. Generando gráficos...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Gráfico 1: Correlaciones por estado
    nombres = list(estados_bell.keys())
    correlaciones = [sim.calcular_correlacion(estados_bell[n]) for n in nombres]
    
    axes[0, 0].bar(nombres, correlaciones, color=['red', 'blue', 'green', 'orange'])
    axes[0, 0].set_ylabel('Correlación <Z1 Z2>')
    axes[0, 0].set_title('Correlaciones ZZ en Estados de Bell')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Desigualdad CHSH
    s_valores = [sim.calcular_desigualdad_bell(estados_bell[n]) for n in nombres]
    
    axes[0, 1].bar(nombres, s_valores, color=['red', 'blue', 'green', 'orange'])
    axes[0, 1].axhline(y=2, color='k', linestyle='--', label='Límite clásico (2)')
    axes[0, 1].axhline(y=2*np.sqrt(2), color='r', linestyle='--', label='Límite Tsirelson (2√2)')
    axes[0, 1].set_ylabel('S')
    axes[0, 1].set_title('Desigualdad de CHSH')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Medidas correlacionadas
    axes[1, 0].scatter(resultados_q1, resultados_q2, alpha=0.5, s=30)
    axes[1, 0].set_xlabel('Qubit 1')
    axes[1, 0].set_ylabel('Qubit 2')
    axes[1, 0].set_title(f'Medidas Correlacionadas (Correlación: {correlacion_medida:.3f})')
    axes[1, 0].set_xlim(-0.5, 1.5)
    axes[1, 0].set_ylim(-0.5, 1.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Histograma de coincidencias
    coincidencias = np.array(resultados_q1) == np.array(resultados_q2)
    no_coincidencias = ~coincidencias
    
    axes[1, 1].bar(['Coincidencias', 'No-coincidencias'], 
                  [coincidencias.sum(), no_coincidencias.sum()],
                  color=['green', 'red'])
    axes[1, 1].set_ylabel('Número de eventos')
    axes[1, 1].set_title('Estadística de Medidas')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('qubits_entrelazados.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reporte
    print("\n7. Generando reporte...")
    reporte = {
        'proyecto': 'Qubits Entrelazados',
        'estados_bell': {
            'definiciones': {
                'Φ+': '(|00> + |11>) / √2',
                'Φ-': '(|00> - |11>) / √2',
                'Ψ+': '(|01> + |10>) / √2',
                'Ψ-': '(|01> - |10>) / √2'
            }
        },
        'correlaciones': {nombres[i]: correlaciones[i] for i in range(len(nombres))},
        'desigualdad_chsh': {nombres[i]: s_valores[i] for i in range(len(nombres))},
        'experimento_medidas': {
            'num_medidas': num_medidas,
            'correlacion_observada': float(correlacion_medida)
        }
    }
    
    with open('REPORTE_ENTRELAZADOS.json', 'w') as f:
        json.dump(reporte, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    main()
