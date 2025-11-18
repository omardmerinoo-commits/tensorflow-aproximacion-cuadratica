"""
Script de demostración para simulaciones de qubit.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulador_qubit import SimuladorQubit
from pathlib import Path
import json

plt.switch_backend('Agg')


def main():
    """Ejecuta demostraciones de simulaciones de qubit."""
    
    print("=" * 70)
    print("PROYECTO 3: SIMULADOR DE QUBIT")
    print("=" * 70)
    
    # Crear directorio de resultados
    output_dir = Path('resultados_simulacion')
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar simulador
    sim = SimuladorQubit(seed=42)
    
    # ==================== 1. ESFERA DE BLOCH ====================
    print("\n[1/4] Visualizando la Esfera de Bloch...")
    
    # Estados importantes
    estados = {
        '|0⟩': sim.estado_inicial('0'),
        '|1⟩': sim.estado_inicial('1'),
        '|+⟩': sim.estado_inicial('+'),
        '|-⟩': sim.estado_inicial('-'),
        '|+i⟩': sim.estado_inicial('+i'),
        '|-i⟩': sim.estado_inicial('-i'),
    }
    
    fig = plt.figure(figsize=(14, 8))
    
    for idx, (nombre, estado) in enumerate(estados.items(), 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        
        # Esfera
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_esfera = np.outer(np.cos(u), np.sin(v))
        y_esfera = np.outer(np.sin(u), np.sin(v))
        z_esfera = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_esfera, y_esfera, z_esfera, alpha=0.2, color='cyan')
        
        # Ejes
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='red', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1.2, 0, color='green', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='blue', arrow_length_ratio=0.1)
        
        # Vector de estado
        ax.quiver(0, 0, 0, estado.x, estado.y, estado.z, 
                 color='black', arrow_length_ratio=0.1, linewidth=3)
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_title(f"{nombre}\n(ρ={estado.pureza:.3f})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'esfera_bloch_estados.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: esfera_bloch_estados.png")
    plt.close()
    
    # ==================== 2. OSCILACIONES DE RABI ====================
    print("\n[2/4] Simulando Oscilaciones de Rabi...")
    
    estado_inicial = sim.estado_inicial('0')
    duraciones, probs = sim.experimento_rabi(estado_inicial, amplitud=1.0, duracion_max=4*np.pi)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(duraciones, probs, linewidth=2.5, color='#3498db')
    ax.fill_between(duraciones, probs, alpha=0.3, color='#3498db')
    ax.axhline(y=0.5, color='red', linestyle='--', label='50% (Máximo)', linewidth=2)
    ax.set_xlabel('Duración del Pulso (rad)', fontsize=12)
    ax.set_ylabel('Probabilidad de Excitación', fontsize=12)
    ax.set_title('Oscilaciones de Rabi: |0⟩ → |1⟩', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'oscilaciones_rabi.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: oscilaciones_rabi.png")
    plt.close()
    
    # ==================== 3. DECOHERENCIA ====================
    print("\n[3/4] Simulando Decoherencia...")
    
    estado_inicial = sim.estado_inicial('+')
    tiempos_total, mag_t2 = [], []
    t2 = 2.0
    
    for t in np.linspace(0, 10*t2, 200):
        estado_dec = sim.decoherencia_t2(estado_inicial, t, t2)
        tiempos_total.append(t)
        mag_t2.append(estado_dec.magnetizacion)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Decoherencia T2
    ax1.plot(tiempos_total, mag_t2, linewidth=2.5, label='Magnetización', color='#e74c3c')
    ax1.fill_between(tiempos_total, mag_t2, alpha=0.3, color='#e74c3c')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Tiempo (s)', fontsize=12)
    ax1.set_ylabel('Magnetización', fontsize=12)
    ax1.set_title('Decoherencia T2 (Defasing)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Trayectoria en Bloch
    estados_trayectoria = []
    for t in np.linspace(0, 5*t2, 50):
        estado = sim.decoherencia_t2(estado_inicial, t, t2)
        estados_trayectoria.append(estado)
    
    xs = [e.x for e in estados_trayectoria]
    ys = [e.y for e in estados_trayectoria]
    zs = [e.z for e in estados_trayectoria]
    
    ax2 = plt.subplot(122, projection='3d')
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_esfera = np.outer(np.cos(u), np.sin(v))
    y_esfera = np.outer(np.sin(u), np.sin(v))
    z_esfera = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x_esfera, y_esfera, z_esfera, alpha=0.2)
    ax2.plot(xs, ys, zs, 'o-', color='#e74c3c', linewidth=2, markersize=4)
    ax2.set_title('Trayectoria: Decoherencia T2', fontsize=13, fontweight='bold')
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decoherencia.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: decoherencia.png")
    plt.close()
    
    # ==================== 4. ECHO DE SPIN ====================
    print("\n[4/4] Simulando Echo de Spin (Hahn Echo)...")
    
    estado_inicial = sim.estado_inicial('+')
    tiempos, mag_libre, mag_echo = sim.experimento_echo(
        estado_inicial, t1=5.0, t2=1.0, tiempos_total=np.linspace(0, 20, 200)
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(tiempos, mag_libre, linewidth=2.5, label='Sin Echo', color='#e74c3c')
    ax.plot(tiempos, mag_echo, linewidth=2.5, label='Con Echo', color='#2ecc71')
    ax.fill_between(tiempos, mag_libre, alpha=0.2, color='#e74c3c')
    ax.fill_between(tiempos, mag_echo, alpha=0.2, color='#2ecc71')
    
    ax.set_xlabel('Tiempo Total (s)', fontsize=12)
    ax.set_ylabel('Magnetización', fontsize=12)
    ax.set_title('Experimento de Hahn Echo: Corrección de Defasing', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hahn_echo.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: hahn_echo.png")
    plt.close()
    
    # ==================== GUARDADO DE RESULTADOS ====================
    print("\n[EXTRA] Guardando resultados...")
    
    resultados = {
        'experimento': 'Simulaciones de Qubit',
        'fecha': '2025-11-18',
        'resultados': {
            'rabi_max_prob': float(np.max(probs)),
            'rabi_min_prob': float(np.min(probs)),
            'decoherencia_inicial': float(mag_t2[0]),
            'decoherencia_final': float(mag_t2[-1]),
            'echo_mejora': float(np.max(mag_echo) - np.max(mag_libre)),
        }
    }
    
    with open(output_dir / 'resultados.json', 'w') as f:
        json.dump(resultados, f, indent=4)
    print("✓ Guardado: resultados.json")
    
    # ==================== RESUMEN ====================
    print("\n" + "=" * 70)
    print("SIMULACIONES COMPLETADAS")
    print("=" * 70)
    print(f"Resultados guardados en: {output_dir.absolute()}")
    print("  ✓ esfera_bloch_estados.png")
    print("  ✓ oscilaciones_rabi.png")
    print("  ✓ decoherencia.png")
    print("  ✓ hahn_echo.png")
    print("  ✓ resultados.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
