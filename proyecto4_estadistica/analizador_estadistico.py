"""
Sistema profesional de análisis estadístico para datasets experimentales.

Proporciona:
- Estadísticas descriptivas completas
- Tests estadísticos (t-test, ANOVA, etc.)
- Visualizaciones automáticas
- Detección de outliers
- Análisis de distribuciones
- Caching de resultados
- Exportación de reportes
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import asdict
import hashlib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class AnalizadorEstadistico:
    """Analizador estadístico profesional."""
    
    def __init__(self, seed: int = 42):
        """Inicializa el analizador."""
        np.random.seed(seed)
        self.cache = {}
        self.historial = []
    
    # ==================== ESTADÍSTICAS DESCRIPTIVAS ====================
    
    def estadisticas_basicas(self, datos: np.ndarray) -> Dict:
        """
        Calcula estadísticas descriptivas básicas.
        
        Args:
            datos: Array de datos
            
        Returns:
            Diccionario con estadísticas
        """
        datos = np.array(datos, dtype=float)
        
        return {
            'n': int(len(datos)),
            'media': float(np.mean(datos)),
            'mediana': float(np.median(datos)),
            'moda': float(stats.mode(datos, keepdims=True).mode[0]) if len(datos) > 0 else None,
            'std': float(np.std(datos, ddof=1)) if len(datos) > 1 else 0,
            'varianza': float(np.var(datos, ddof=1)) if len(datos) > 1 else 0,
            'minimo': float(np.min(datos)),
            'maximo': float(np.max(datos)),
            'rango': float(np.max(datos) - np.min(datos)),
            'q1': float(np.percentile(datos, 25)),
            'q2': float(np.percentile(datos, 50)),
            'q3': float(np.percentile(datos, 75)),
            'iqr': float(np.percentile(datos, 75) - np.percentile(datos, 25)),
            'sesgo': float(stats.skew(datos)) if len(datos) > 2 else 0,
            'curtosis': float(stats.kurtosis(datos)) if len(datos) > 3 else 0,
            'coef_variacion': float(np.std(datos, ddof=1) / np.mean(datos) * 100) if np.mean(datos) != 0 else 0,
        }
    
    def intervalo_confianza(self, datos: np.ndarray, confianza: float = 0.95) -> Tuple[float, float]:
        """
        Calcula intervalo de confianza para la media.
        
        Args:
            datos: Array de datos
            confianza: Nivel de confianza (por defecto 95%)
            
        Returns:
            Tupla (inferior, superior)
        """
        datos = np.array(datos, dtype=float)
        n = len(datos)
        media = np.mean(datos)
        error_std = stats.sem(datos)
        
        # Valor crítico t
        alpha = 1 - confianza
        t_critico = stats.t.ppf(1 - alpha/2, n-1)
        
        margen_error = t_critico * error_std
        
        return (float(media - margen_error), float(media + margen_error))
    
    def deteccion_outliers(self, datos: np.ndarray, metodo: str = 'iqr', 
                          umbral: float = 1.5) -> Dict:
        """
        Detecta outliers en los datos.
        
        Args:
            datos: Array de datos
            metodo: 'iqr' (rango intercuartílico) o 'zscore'
            umbral: Umbral para detección
            
        Returns:
            Diccionario con información de outliers
        """
        datos = np.array(datos, dtype=float)
        
        if metodo == 'iqr':
            q1 = np.percentile(datos, 25)
            q3 = np.percentile(datos, 75)
            iqr = q3 - q1
            
            limite_inf = q1 - umbral * iqr
            limite_sup = q3 + umbral * iqr
            
            outliers = datos[(datos < limite_inf) | (datos > limite_sup)]
            indices = np.where((datos < limite_inf) | (datos > limite_sup))[0]
            
        elif metodo == 'zscore':
            z_scores = np.abs(stats.zscore(datos))
            outliers = datos[z_scores > umbral]
            indices = np.where(z_scores > umbral)[0]
        
        else:
            raise ValueError(f"Método {metodo} no reconocido")
        
        return {
            'metodo': metodo,
            'num_outliers': int(len(outliers)),
            'porcentaje': float(len(outliers) / len(datos) * 100),
            'outliers': list(map(float, outliers)),
            'indices': list(map(int, indices)),
        }
    
    # ==================== TESTS ESTADÍSTICOS ====================
    
    def test_normalidad(self, datos: np.ndarray) -> Dict:
        """
        Realiza tests de normalidad.
        
        Args:
            datos: Array de datos
            
        Returns:
            Diccionario con resultados de tests
        """
        datos = np.array(datos, dtype=float)
        
        # Test de Shapiro-Wilk
        stat_shapiro, p_shapiro = stats.shapiro(datos)
        
        # Test de Kolmogorov-Smirnov
        stat_ks, p_ks = stats.kstest(datos, 'norm', args=(np.mean(datos), np.std(datos)))
        
        # Test de Anderson-Darling
        resultado_anderson = stats.anderson(datos)
        
        return {
            'shapiro_wilk': {
                'estadistico': float(stat_shapiro),
                'p_valor': float(p_shapiro),
                'es_normal': bool(p_shapiro > 0.05)
            },
            'kolmogorov_smirnov': {
                'estadistico': float(stat_ks),
                'p_valor': float(p_ks),
                'es_normal': bool(p_ks > 0.05)
            },
            'anderson_darling': {
                'estadistico': float(resultado_anderson.statistic),
                'nivel_significancia': float(resultado_anderson.critical_values[2]),
                'es_normal': bool(resultado_anderson.statistic < resultado_anderson.critical_values[2])
            }
        }
    
    def test_t_independiente(self, datos1: np.ndarray, datos2: np.ndarray) -> Dict:
        """
        Test t de Student para muestras independientes.
        
        Args:
            datos1: Primera muestra
            datos2: Segunda muestra
            
        Returns:
            Resultados del test
        """
        datos1 = np.array(datos1, dtype=float)
        datos2 = np.array(datos2, dtype=float)
        
        # Test de igualdad de varianzas (Levene)
        stat_levene, p_levene = stats.levene(datos1, datos2)
        varianzas_iguales = p_levene > 0.05
        
        # T-test
        stat_t, p_valor = stats.ttest_ind(datos1, datos2, equal_var=varianzas_iguales)
        
        # Tamaño del efecto (Cohen's d)
        media1, media2 = np.mean(datos1), np.mean(datos2)
        std1, std2 = np.std(datos1, ddof=1), np.std(datos2, ddof=1)
        n1, n2 = len(datos1), len(datos2)
        
        std_combined = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        cohens_d = (media1 - media2) / std_combined if std_combined > 0 else 0
        
        return {
            'levene_test': {
                'estadistico': float(stat_levene),
                'p_valor': float(p_levene),
                'varianzas_iguales': bool(varianzas_iguales)
            },
            't_test': {
                'estadistico': float(stat_t),
                'p_valor': float(p_valor),
                'significativo': bool(p_valor < 0.05)
            },
            'cohens_d': float(cohens_d),
            'medias': {
                'muestra1': float(media1),
                'muestra2': float(media2),
                'diferencia': float(media1 - media2)
            }
        }
    
    def test_anova(self, *muestras) -> Dict:
        """
        ANOVA de una vía para comparar múltiples grupos.
        
        Args:
            *muestras: Múltiples arrays de datos
            
        Returns:
            Resultados del ANOVA
        """
        # Convertir a float
        muestras = [np.array(m, dtype=float) for m in muestras]
        
        # ANOVA
        stat_f, p_valor = stats.f_oneway(*muestras)
        
        # Eta-squared (tamaño del efecto)
        todas_datos = np.concatenate(muestras)
        media_global = np.mean(todas_datos)
        
        ss_between = sum(len(m) * (np.mean(m) - media_global)**2 for m in muestras)
        ss_total = sum((x - media_global)**2 for x in todas_datos)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'f_estadistico': float(stat_f),
            'p_valor': float(p_valor),
            'significativo': bool(p_valor < 0.05),
            'eta_squared': float(eta_squared),
            'num_grupos': len(muestras),
            'medias_grupos': [float(np.mean(m)) for m in muestras],
            'tamanios_grupos': [len(m) for m in muestras]
        }
    
    def test_correlacion(self, x: np.ndarray, y: np.ndarray, 
                        metodo: str = 'pearson') -> Dict:
        """
        Test de correlación entre dos variables.
        
        Args:
            x: Primera variable
            y: Segunda variable
            metodo: 'pearson', 'spearman', 'kendall'
            
        Returns:
            Resultados de correlación
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        
        if metodo == 'pearson':
            corr, p_valor = stats.pearsonr(x, y)
        elif metodo == 'spearman':
            corr, p_valor = stats.spearmanr(x, y)
        elif metodo == 'kendall':
            corr, p_valor = stats.kendalltau(x, y)
        else:
            raise ValueError(f"Método {metodo} no reconocido")
        
        # R-squared
        r_squared = corr**2
        
        return {
            'metodo': metodo,
            'correlacion': float(corr),
            'r_squared': float(r_squared),
            'p_valor': float(p_valor),
            'significativo': bool(p_valor < 0.05),
            'interpretacion': self._interpretar_correlacion(corr)
        }
    
    @staticmethod
    def _interpretar_correlacion(r: float) -> str:
        """Interpreta el coeficiente de correlación."""
        r_abs = abs(r)
        if r_abs < 0.3:
            return "débil"
        elif r_abs < 0.7:
            return "moderada"
        else:
            return "fuerte"
    
    # ==================== ANÁLISIS DE DISTRIBUCIONES ====================
    
    def ajuste_distribucion(self, datos: np.ndarray) -> Dict:
        """
        Prueba ajuste a varias distribuciones comunes.
        
        Args:
            datos: Array de datos
            
        Returns:
            Diccionario con KS estadístico para cada distribución
        """
        datos = np.array(datos, dtype=float)
        
        resultados = {}
        
        # Normal
        stat, p = stats.kstest(datos, 'norm', args=(np.mean(datos), np.std(datos)))
        resultados['normal'] = {'ks_stat': float(stat), 'p_valor': float(p)}
        
        # Log-normal
        try:
            s, loc, scale = stats.lognorm.fit(datos)
            stat, p = stats.kstest(datos, lambda x: stats.lognorm.cdf(x, s, loc, scale))
            resultados['lognormal'] = {'ks_stat': float(stat), 'p_valor': float(p)}
        except:
            resultados['lognormal'] = {'ks_stat': np.inf, 'p_valor': 0}
        
        # Exponencial
        try:
            loc, scale = stats.expon.fit(datos)
            stat, p = stats.kstest(datos, lambda x: stats.expon.cdf(x, loc, scale))
            resultados['exponencial'] = {'ks_stat': float(stat), 'p_valor': float(p)}
        except:
            resultados['exponencial'] = {'ks_stat': np.inf, 'p_valor': 0}
        
        # Uniforme
        try:
            loc, scale = stats.uniform.fit(datos)
            stat, p = stats.kstest(datos, lambda x: stats.uniform.cdf(x, loc, scale))
            resultados['uniforme'] = {'ks_stat': float(stat), 'p_valor': float(p)}
        except:
            resultados['uniforme'] = {'ks_stat': np.inf, 'p_valor': 0}
        
        return resultados
    
    # ==================== VISUALIZACIONES ====================
    
    def visualizar_analisis_completo(self, datos: np.ndarray, nombre: str = "Análisis",
                                    archivo_salida: str = None):
        """
        Crea visualización completa de los datos.
        
        Args:
            datos: Array de datos
            nombre: Nombre del análisis
            archivo_salida: Ruta para guardar
        """
        datos = np.array(datos, dtype=float)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histograma
        axes[0, 0].hist(datos, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(datos), color='red', linestyle='--', label=f'Media: {np.mean(datos):.2f}')
        axes[0, 0].axvline(np.median(datos), color='green', linestyle='--', label=f'Mediana: {np.median(datos):.2f}')
        axes[0, 0].set_title('Histograma de Frecuencias')
        axes[0, 0].set_xlabel('Valor')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(datos, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normalidad)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot(datos, vert=True)
        axes[1, 0].set_title('Diagrama de Caja')
        axes[1, 0].set_ylabel('Valor')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Densidad
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(datos)
        x_range = np.linspace(datos.min(), datos.max(), 100)
        axes[1, 1].plot(x_range, kde(x_range), 'b-', linewidth=2)
        axes[1, 1].fill_between(x_range, kde(x_range), alpha=0.3)
        axes[1, 1].hist(datos, bins=30, density=True, alpha=0.3, color='gray')
        axes[1, 1].set_title('Estimación de Densidad')
        axes[1, 1].set_xlabel('Valor')
        axes[1, 1].set_ylabel('Densidad')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{nombre} - Análisis Estadístico Completo', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            print(f"Guardado: {archivo_salida}")
        else:
            plt.show()
        
        plt.close()
    
    # ==================== EXPORTACIÓN DE REPORTES ====================
    
    def generar_reporte(self, datos: np.ndarray, nombre: str = "Análisis") -> Dict:
        """
        Genera un reporte estadístico completo.
        
        Args:
            datos: Array de datos
            nombre: Nombre del análisis
            
        Returns:
            Diccionario con reporte completo
        """
        datos = np.array(datos, dtype=float)
        
        reporte = {
            'titulo': f'Reporte Estadístico: {nombre}',
            'fecha': datetime.now().isoformat(),
            'num_observaciones': len(datos),
            'estadisticas_basicas': self.estadisticas_basicas(datos),
            'intervalo_confianza': {
                'inferior': self.intervalo_confianza(datos)[0],
                'superior': self.intervalo_confianza(datos)[1]
            },
            'outliers': self.deteccion_outliers(datos),
            'tests_normalidad': self.test_normalidad(datos),
            'distribucion_ajuste': self.ajuste_distribucion(datos),
        }
        
        return reporte
    
    def exportar_reporte_json(self, reporte: Dict, archivo: str):
        """Exporta reporte a JSON."""
        with open(archivo, 'w') as f:
            json.dump(reporte, f, indent=4, ensure_ascii=False)
        print(f"Reporte guardado en: {archivo}")
    
    def exportar_reporte_texto(self, reporte: Dict, archivo: str):
        """Exporta reporte a texto."""
        with open(archivo, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"{reporte['titulo']}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Fecha: {reporte['fecha']}\n")
            f.write(f"Observaciones: {reporte['num_observaciones']}\n\n")
            
            f.write("ESTADÍSTICAS DESCRIPTIVAS\n")
            f.write("-" * 70 + "\n")
            for k, v in reporte['estadisticas_basicas'].items():
                if v is not None:
                    f.write(f"  {k:.<30} {v:.6f}\n")
            
            f.write("\nINTERVALO DE CONFIANZA (95%)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  [inferior, superior] = [{reporte['intervalo_confianza']['inferior']:.6f}, {reporte['intervalo_confianza']['superior']:.6f}]\n")
            
            f.write("\nOUTLIERS\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total: {reporte['outliers']['num_outliers']} ({reporte['outliers']['porcentaje']:.2f}%)\n")
        
        print(f"Reporte guardado en: {archivo}")
