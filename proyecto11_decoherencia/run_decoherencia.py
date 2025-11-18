"""Script ejecución Proyecto 11 - Decoherencia."""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import json
from simulador_decoherencia import SimuladorDecoherencia


def main():
    print("=" * 70)
    print("SIMULACION DE DECOHERENCIA EN QUBITS")
    print("=" * 70)
    
    # Parámetros
    T1 = 2.0
    T2 = 1.0
    
    print(f"\n1. Parámetros:")
    print(f"   T1 (relajación): {T1}")
    print(f"   T2 (dephasing): {T2}")
    
    # Crear simulador
    sim = SimuladorDecoherencia(T1=T1, T2=T2)
    
    # Parte 1: Decoherencia desde |+>
    print(f"\n2. Simulando decoherencia desde |+>...")
    estado_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    tiempos, x_vals, z_vals = sim.simular_decoherencia(
        estado_plus, tiempo_max=10.0, num_puntos=100
    )
    print(f"   <σx> inicial: {x_vals[0]:.4f}, final: {x_vals[-1]:.4f}")
    print(f"   <σz> inicial: {z_vals[0]:.4f}, final: {z_vals[-1]:.4f}")
    
    # Parte 2: Decoherencia desde |0>
    print(f"\n3. Simulando decoherencia desde |0>...")
    estado_0 = qt.basis(2, 0)
    tiempos2, x_vals2, z_vals2 = sim.simular_decoherencia(
        estado_0, tiempo_max=10.0, num_puntos=100
    )
    print(f"   <σx> inicial: {x_vals2[0]:.4f}, final: {x_vals2[-1]:.4f}")
    print(f"   <σz> inicial: {z_vals2[0]:.4f}, final: {z_vals2[-1]:.4f}")
    
    # Parte 3: Eco de Hahn
    print(f"\n4. Simulando Eco de Hahn...")
    tiempos_hahn, coherencia = sim.simular_eco_hahn(
        estado_plus, tiempo_max=10.0, num_puntos=100
    )
    print(f"   Coherencia inicial: {coherencia[0]:.4f}")
    print(f"   Coherencia final: {coherencia[-1]:.4f}")
    
    # Gráficos
    print(f"\n5. Generando gráficos...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Decoherencia desde |+>
    axes[0, 0].plot(tiempos, x_vals, 'b-', label='<σx>', linewidth=2)
    axes[0, 0].plot(tiempos, z_vals, 'r-', label='<σz>', linewidth=2)
    axes[0, 0].set_xlabel('Tiempo')
    axes[0, 0].set_ylabel('Valor esperado')
    axes[0, 0].set_title('Decoherencia desde |+>')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Decoherencia desde |0>
    axes[0, 1].plot(tiempos2, x_vals2, 'g-', label='<σx>', linewidth=2)
    axes[0, 1].plot(tiempos2, z_vals2, 'purple', label='<σz>', linewidth=2)
    axes[0, 1].set_xlabel('Tiempo')
    axes[0, 1].set_ylabel('Valor esperado')
    axes[0, 1].set_title('Decoherencia desde |0>')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Eco de Hahn
    axes[1, 0].plot(tiempos_hahn, coherencia, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Tiempo')
    axes[1, 0].set_ylabel('Coherencia |<σx>|')
    axes[1, 0].set_title('Eco de Hahn (Refocalización)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Comparación
    axes[1, 1].plot(tiempos, np.abs(x_vals), 'b-', label='|<σx>| desde |+>', linewidth=2)
    axes[1, 1].plot(tiempos_hahn, coherencia, 'orange', label='Eco Hahn', linewidth=2)
    axes[1, 1].set_xlabel('Tiempo')
    axes[1, 1].set_ylabel('Coherencia')
    axes[1, 1].set_title('Efecto del Eco de Hahn')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decoherencia.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reporte
    print(f"\n6. Generando reporte...")
    reporte = {
        'proyecto': 'Simulación de Decoherencia',
        'parametros': {'T1': T1, 'T2': T2},
        'tasas': {
            'gamma_T1': float(sim.tasa_relajacion_t1()),
            'gamma_T2': float(sim.tasa_dephasing_t2())
        },
        'resultados': {
            'decoherencia_plus': {
                '<σx>_inicial': float(x_vals[0]),
                '<σx>_final': float(x_vals[-1])
            },
            'eco_hahn': {
                'coherencia_inicial': float(coherencia[0]),
                'coherencia_final': float(coherencia[-1])
            }
        }
    }
    
    with open('REPORTE_DECOHERENCIA.json', 'w') as f:
        json.dump(reporte, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    main()
