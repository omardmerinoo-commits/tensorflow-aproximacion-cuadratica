"""
Tests para el analizador estadístico.
"""

import pytest
import numpy as np
from analizador_estadistico import AnalizadorEstadistico


class TestEstadisticasBasicas:
    """Tests para estadísticas descriptivas."""
    
    def setup_method(self):
        """Configuración."""
        self.analizador = AnalizadorEstadistico(seed=42)
    
    def test_estadisticas_completas(self):
        """Verifica cálculo de estadísticas completas."""
        datos = np.array([1, 2, 3, 4, 5])
        stats = self.analizador.estadisticas_basicas(datos)
        
        assert stats['n'] == 5
        assert np.isclose(stats['media'], 3.0)
        assert np.isclose(stats['mediana'], 3.0)
    
    def test_media_correcta(self):
        """Verifica cálculo de media."""
        datos = np.array([10, 20, 30])
        stats = self.analizador.estadisticas_basicas(datos)
        assert np.isclose(stats['media'], 20.0)
    
    def test_std_correcta(self):
        """Verifica cálculo de desviación estándar."""
        datos = np.array([1, 2, 3, 4, 5])
        stats = self.analizador.estadisticas_basicas(datos)
        esperado = np.std([1, 2, 3, 4, 5], ddof=1)
        assert np.isclose(stats['std'], esperado)
    
    def test_percentiles(self):
        """Verifica cálculo de percentiles."""
        datos = np.arange(100)
        stats = self.analizador.estadisticas_basicas(datos)
        
        assert np.isclose(stats['q1'], 24.75)
        assert np.isclose(stats['q2'], 49.5)
        assert np.isclose(stats['q3'], 74.25)


class TestIntervaloConfianza:
    """Tests para intervalo de confianza."""
    
    def setup_method(self):
        """Configuración."""
        self.analizador = AnalizadorEstadistico(seed=42)
    
    def test_intervalo_contiene_media(self):
        """Verifica que el intervalo contiene la media."""
        datos = np.random.normal(100, 15, 1000)
        inf, sup = self.analizador.intervalo_confianza(datos, confianza=0.95)
        media = np.mean(datos)
        
        assert inf < media < sup
    
    def test_intervalo_mayor_para_menor_confianza(self):
        """Verifica que menor confianza genera intervalo menor."""
        datos = np.random.normal(100, 15, 1000)
        
        inf_90, sup_90 = self.analizador.intervalo_confianza(datos, confianza=0.90)
        inf_95, sup_95 = self.analizador.intervalo_confianza(datos, confianza=0.95)
        
        # El intervalo de 95% debe ser mayor
        assert (sup_95 - inf_95) > (sup_90 - inf_90)


class TestDeteccionOutliers:
    """Tests para detección de outliers."""
    
    def setup_method(self):
        """Configuración."""
        self.analizador = AnalizadorEstadistico(seed=42)
    
    def test_detecta_outliers_iqr(self):
        """Verifica detección de outliers con IQR."""
        datos = np.array([1, 2, 3, 4, 5, 100])
        result = self.analizador.deteccion_outliers(datos, metodo='iqr')
        
        assert result['num_outliers'] > 0
        assert 100 in result['outliers']
    
    def test_detecta_outliers_zscore(self):
        """Verifica detección con Z-score."""
        datos = np.array([1, 2, 3, 4, 5, 100])
        result = self.analizador.deteccion_outliers(datos, metodo='zscore', umbral=2)
        
        assert result['num_outliers'] > 0


class TestTestsEstadisticos:
    """Tests para tests estadísticos."""
    
    def setup_method(self):
        """Configuración."""
        self.analizador = AnalizadorEstadistico(seed=42)
    
    def test_test_normalidad(self):
        """Verifica test de normalidad."""
        datos = np.random.normal(0, 1, 100)
        result = self.analizador.test_normalidad(datos)
        
        assert 'shapiro_wilk' in result
        assert 'kolmogorov_smirnov' in result
        assert 'anderson_darling' in result
    
    def test_ttest_muestras_iguales(self):
        """Verifica t-test con muestras iguales."""
        datos1 = np.random.normal(0, 1, 100)
        datos2 = np.random.normal(0, 1, 100)
        
        result = self.analizador.test_t_independiente(datos1, datos2)
        
        assert 't_test' in result
        assert 'cohens_d' in result
        assert 0 <= result['t_test']['p_valor'] <= 1
    
    def test_ttest_muestras_diferentes(self):
        """Verifica t-test con muestras significativamente diferentes."""
        datos1 = np.random.normal(0, 1, 100)
        datos2 = np.random.normal(5, 1, 100)
        
        result = self.analizador.test_t_independiente(datos1, datos2)
        
        # Debe ser significativo
        assert result['t_test']['p_valor'] < 0.05
    
    def test_anova_un_grupo(self):
        """Verifica ANOVA con 1 grupo."""
        grupo = np.random.normal(0, 1, 50)
        result = self.analizador.test_anova(grupo)
        
        assert result['num_grupos'] == 1
    
    def test_anova_multiples_grupos(self):
        """Verifica ANOVA con múltiples grupos."""
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(5, 1, 50)
        g3 = np.random.normal(10, 1, 50)
        
        result = self.analizador.test_anova(g1, g2, g3)
        
        assert result['num_grupos'] == 3
        # Debe ser muy significativo
        assert result['p_valor'] < 0.05
    
    def test_correlacion_perfecta(self):
        """Verifica correlación perfecta."""
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x  # Correlación perfecta
        
        result = self.analizador.test_correlacion(x, y, metodo='pearson')
        
        assert np.isclose(result['correlacion'], 1.0, atol=1e-10)
        assert np.isclose(result['r_squared'], 1.0, atol=1e-10)
    
    def test_correlacion_nula(self):
        """Verifica sin correlación."""
        np.random.seed(42)
        x = np.random.normal(size=100)
        y = np.random.normal(size=100)
        
        result = self.analizador.test_correlacion(x, y, metodo='pearson')
        
        # Correlación debe ser cercana a 0
        assert abs(result['correlacion']) < 0.3


class TestAjusteDistribucion:
    """Tests para ajuste de distribuciones."""
    
    def setup_method(self):
        """Configuración."""
        self.analizador = AnalizadorEstadistico(seed=42)
    
    def test_ajuste_distribucion_normal(self):
        """Verifica ajuste a distribución normal."""
        datos = np.random.normal(0, 1, 1000)
        result = self.analizador.ajuste_distribucion(datos)
        
        assert 'normal' in result
        # Normal debe tener baja distancia KS
        assert result['normal']['ks_stat'] < 0.5
    
    def test_ajuste_distribucion_exponencial(self):
        """Verifica ajuste a distribución exponencial."""
        datos = np.random.exponential(2, 1000)
        result = self.analizador.ajuste_distribucion(datos)
        
        assert 'exponencial' in result
        # Exponencial debe ser el mejor ajuste
        ks_exp = result['exponencial']['ks_stat']
        ks_norm = result['normal']['ks_stat']
        assert ks_exp < ks_norm


class TestGenerarReporte:
    """Tests para generación de reportes."""
    
    def setup_method(self):
        """Configuración."""
        self.analizador = AnalizadorEstadistico(seed=42)
    
    def test_generar_reporte_completo(self):
        """Verifica generación de reporte completo."""
        datos = np.random.normal(100, 15, 500)
        reporte = self.analizador.generar_reporte(datos, "Test")
        
        assert 'titulo' in reporte
        assert 'fecha' in reporte
        assert 'estadisticas_basicas' in reporte
        assert 'tests_normalidad' in reporte
        assert 'distribucion_ajuste' in reporte
    
    def test_reporte_contiene_estadisticas(self):
        """Verifica que reporte contiene estadísticas."""
        datos = np.random.normal(100, 15, 500)
        reporte = self.analizador.generar_reporte(datos)
        
        stats = reporte['estadisticas_basicas']
        assert stats['n'] == 500
        assert 'media' in stats
        assert 'std' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
