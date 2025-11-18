"""
Script de demostración para análisis estadístico.
"""

import numpy as np
import matplotlib.pyplot as plt
from analizador_estadistico import AnalizadorEstadistico
from pathlib import Path
import json

plt.switch_backend('Agg')


def main():
    """Ejecuta demostraciones de análisis estadístico."""
    
    print("=" * 70)
    print("PROYECTO 4: ANÁLISIS ESTADÍSTICO DE DATASETS")
    print("=" * 70)
    
    # Crear directorio de resultados
    output_dir = Path('resultados_analisis')
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar analizador
    analizador = AnalizadorEstadistico(seed=42)
    
    # ==================== CASO 1: DATOS NORMALES ====================
    print("\n[1/3] Analizando distribución normal...")
    
    datos_normal = np.random.normal(loc=100, scale=15, size=1000)
    
    reporte_normal = analizador.generar_reporte(datos_normal, "Distribución Normal")
    analizador.exportar_reporte_json(reporte_normal, output_dir / 'reporte_normal.json')
    analizador.exportar_reporte_texto(reporte_normal, output_dir / 'reporte_normal.txt')
    analizador.visualizar_analisis_completo(datos_normal, "Datos Normales", 
                                           output_dir / 'grafica_normal.png')
    
    print(f"✓ Estadísticas básicas:")
    print(f"  Media: {reporte_normal['estadisticas_basicas']['media']:.2f}")
    print(f"  Std: {reporte_normal['estadisticas_basicas']['std']:.2f}")
    print(f"  Outliers: {reporte_normal['outliers']['num_outliers']} ({reporte_normal['outliers']['porcentaje']:.2f}%)")
    
    # ==================== CASO 2: DATOS SESGADOS ====================
    print("\n[2/3] Analizando distribución sesgada...")
    
    # Generar datos con sesgo
    datos_sesgados = np.random.exponential(scale=50, size=1000)
    
    reporte_sesgado = analizador.generar_reporte(datos_sesgados, "Distribución Sesgada")
    analizador.exportar_reporte_json(reporte_sesgado, output_dir / 'reporte_sesgado.json')
    analizador.visualizar_analisis_completo(datos_sesgados, "Datos Sesgados", 
                                           output_dir / 'grafica_sesgada.png')
    
    print(f"✓ Estadísticas básicas:")
    print(f"  Sesgo: {reporte_sesgado['estadisticas_basicas']['sesgo']:.4f}")
    print(f"  Curtosis: {reporte_sesgado['estadisticas_basicas']['curtosis']:.4f}")
    
    # ==================== CASO 3: COMPARACIÓN DE GRUPOS ====================
    print("\n[3/3] Realizando tests de comparación...")
    
    # Grupo 1: Antes del tratamiento
    grupo_antes = np.random.normal(loc=50, scale=10, size=100)
    
    # Grupo 2: Después del tratamiento
    grupo_despues = np.random.normal(loc=58, scale=12, size=100)
    
    # T-test
    resultado_ttest = analizador.test_t_independiente(grupo_antes, grupo_despues)
    
    print(f"✓ Test t de Student:")
    print(f"  Media antes: {resultado_ttest['medias']['muestra1']:.2f}")
    print(f"  Media después: {resultado_ttest['medias']['muestra2']:.2f}")
    print(f"  Diferencia: {resultado_ttest['medias']['diferencia']:.2f}")
    print(f"  p-valor: {resultado_ttest['t_test']['p_valor']:.6f}")
    print(f"  Significativo: {'Sí' if resultado_ttest['t_test']['significativo'] else 'No'}")
    print(f"  Cohen's d: {resultado_ttest['cohens_d']:.4f}")
    
    # ANOVA con 3 grupos
    grupo_1 = np.random.normal(loc=50, scale=10, size=100)
    grupo_2 = np.random.normal(loc=60, scale=10, size=100)
    grupo_3 = np.random.normal(loc=70, scale=10, size=100)
    
    resultado_anova = analizador.test_anova(grupo_1, grupo_2, grupo_3)
    
    print(f"\n✓ ANOVA de 3 grupos:")
    print(f"  F-estadístico: {resultado_anova['f_estadistico']:.4f}")
    print(f"  p-valor: {resultado_anova['p_valor']:.6f}")
    print(f"  Medias: {resultado_anova['medias_grupos']}")
    print(f"  Eta-squared: {resultado_anova['eta_squared']:.4f}")
    
    # ==================== CORRELACIÓN ====================
    print("\n[EXTRA] Análisis de correlación...")
    
    # Generar variables correlacionadas
    x = np.random.normal(size=500)
    y = 2 * x + np.random.normal(0, 0.5, size=500)
    
    correlacion = analizador.test_correlacion(x, y, metodo='pearson')
    
    print(f"✓ Correlación de Pearson:")
    print(f"  Correlación: {correlacion['correlacion']:.4f}")
    print(f"  R²: {correlacion['r_squared']:.4f}")
    print(f"  p-valor: {correlacion['p_valor']:.6f}")
    print(f"  Interpretación: {correlacion['interpretacion']}")
    
    # Visualizar correlación
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, alpha=0.5, s=30)
    
    # Recta de regresión
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    ax.plot(x_sorted, p(x_sorted), "r-", linewidth=2, label='Regresión')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Análisis de Correlación (r={correlacion["correlacion"]:.3f})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlacion.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: correlacion.png")
    plt.close()
    
    # ==================== GUARDADO DE RESULTADOS ====================
    print("\n[EXTRA] Compilando resultados finales...")
    
    resultados_finales = {
        'titulo': 'Análisis Estadístico Completo',
        'fecha': '2025-11-18',
        'casos_estudiados': 3,
        'resumen': {
            'distribucion_normal': {
                'media': float(reporte_normal['estadisticas_basicas']['media']),
                'std': float(reporte_normal['estadisticas_basicas']['std']),
                'normalidad_shapiro': reporte_normal['tests_normalidad']['shapiro_wilk']['es_normal']
            },
            'ttest_significativo': resultado_ttest['t_test']['significativo'],
            'anova_significativo': resultado_anova['f_estadistico'],
            'correlacion_pearson': float(correlacion['correlacion'])
        }
    }
    
    with open(output_dir / 'resultados_finales.json', 'w') as f:
        json.dump(resultados_finales, f, indent=4)
    print("✓ Guardado: resultados_finales.json")
    
    # ==================== RESUMEN ====================
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"Resultados guardados en: {output_dir.absolute()}")
    print("  ✓ reporte_normal.json / .txt")
    print("  ✓ reporte_sesgado.json")
    print("  ✓ grafica_normal.png")
    print("  ✓ grafica_sesgada.png")
    print("  ✓ correlacion.png")
    print("  ✓ resultados_finales.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
