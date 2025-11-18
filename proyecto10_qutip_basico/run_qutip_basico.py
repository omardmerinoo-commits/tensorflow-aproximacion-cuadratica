"""Script ejecución Proyecto 10 - QuTiP Básico."""

import numpy as np
import matplotlib.pyplot as plt
import json
from simulador_qutip_basico import SimuladorCuanticoBasico


def main():
    print("=" * 70)
    print("SIMULADOR CUANTICO BASICO CON QuTiP")
    print("=" * 70)
    
    sim = SimuladorCuanticoBasico()
    resultados = {}
    
    # Parte 1: Estados básicos y Bloch
    print("\n1. Estados básicos y esfera de Bloch...")
    bloches = {}
    for nombre in ['|0>', '|1>', '|+>', '|+i>']:
        estado = sim.obtener_estado(nombre)
        x, y, z = sim.calcular_bloch(estado)
        bloches[nombre] = {'x': x, 'y': y, 'z': z}
        print(f"   {nombre}: ({x:.4f}, {y:.4f}, {z:.4f})")
    
    # Parte 2: Operadores de Pauli
    print("\n2. Aplicando operadores de Pauli...")
    estado_0 = sim.obtener_estado('|0>')
    operadores_aplicar = ['X', 'Y', 'Z']
    
    for op_nombre in operadores_aplicar:
        op = sim.obtener_operador(op_nombre)
        estado_nuevo = sim.aplicar_operador(estado_0, op)
        print(f"   Aplicar {op_nombre} a |0>: {estado_nuevo.full().flatten()}")
    
    # Parte 3: Evolución temporal
    print("\n3. Evolución temporal bajo Hamiltoniano...")
    H = sim.crear_hamiltoniano_precesion(frecuencia=1.0)
    estado_inicial = sim.obtener_estado('|0>')
    tiempos, estados_evolucion = sim.evolucionar_temporal(
        estado_inicial, H, tiempo_max=2*np.pi, num_puntos=50
    )
    print(f"   Puntos temporales: {len(tiempos)}")
    
    # Calcular valores esperados de Pauli Z
    z_valores = []
    for estado in estados_evolucion:
        z_valores.append(sim.calcular_bloch(estado)[2])
    
    # Parte 4: Fidelidad
    print("\n4. Fidelidades entre estados...")
    estado1 = sim.obtener_estado('|0>')
    estado2 = sim.obtener_estado('|+>')
    fid = sim.calcular_fidelidad(estado1, estado2)
    print(f"   Fidelidad |0> vs |+>: {fid:.6f}")
    
    # Parte 5: Entropía
    print("\n5. Entropía de von Neumann...")
    for nombre in ['|0>', '|+>']:
        estado = sim.obtener_estado(nombre)
        entropia = sim.calcular_entropia(estado)
        print(f"   {nombre}: {entropia:.6f}")
    
    # Gráfico
    print("\n6. Generando gráficos...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(tiempos, z_valores, 'b-', linewidth=2)
    axes[0].set_xlabel('Tiempo')
    axes[0].set_ylabel('<Z>')
    axes[0].set_title('Evolución de <σz> bajo precesión')
    axes[0].grid(True, alpha=0.3)
    
    bloches_nombres = list(bloches.keys())
    bloches_coords = list(bloches.values())
    xs = [b['x'] for b in bloches_coords]
    ys = [b['y'] for b in bloches_coords]
    zs = [b['z'] for b in bloches_coords]
    
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.scatter(xs, ys, zs, s=100, c='red')
    for i, txt in enumerate(bloches_nombres):
        ax3d.text(xs[i], ys[i], zs[i], txt)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('Esfera de Bloch')
    
    plt.tight_layout()
    plt.savefig('qutip_basico.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reporte
    print("\n7. Generando reporte...")
    reporte = {
        'proyecto': 'Simulador QuTiP Básico',
        'estados': bloches,
        'entropias': {
            '|0>': float(sim.calcular_entropia(sim.obtener_estado('|0>'))),
            '|+>': float(sim.calcular_entropia(sim.obtener_estado('|+>')))
        }
    }
    
    with open('REPORTE_QUTIP.json', 'w') as f:
        json.dump(reporte, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == '__main__':
    main()
