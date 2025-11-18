"""
Tests para el simulador de qubit.
"""

import pytest
import numpy as np
from simulador_qubit import SimuladorQubit, EstadoQubit


class TestEstadoQubit:
    """Tests para la clase EstadoQubit."""
    
    def test_estado_qubit_creacion(self):
        """Verifica creación de un estado."""
        estado = EstadoQubit(x=0, y=0, z=1)
        assert estado.x == 0
        assert estado.y == 0
        assert estado.z == 1
    
    def test_magnetizacion(self):
        """Verifica cálculo de magnetización."""
        estado = EstadoQubit(x=1, y=0, z=0)
        assert np.isclose(estado.magnetizacion, 1.0)
    
    def test_pureza_puro(self):
        """Verifica pureza de estado puro."""
        estado = EstadoQubit(x=0, y=0, z=1)
        assert np.isclose(estado.pureza, 1.0)
    
    def test_pureza_mezcla(self):
        """Verifica pureza de estado mixto."""
        estado = EstadoQubit(x=0, y=0, z=0)
        assert np.isclose(estado.pureza, 0.5)
    
    def test_punto_bloch(self):
        """Verifica retorno de punto de Bloch."""
        estado = EstadoQubit(x=0.5, y=0.5, z=0.7)
        punto = estado.punto_bloch
        assert punto == (0.5, 0.5, 0.7)
    
    def test_to_dict(self):
        """Verifica conversión a diccionario."""
        estado = EstadoQubit(x=0.3, y=0.4, z=0.8)
        d = estado.to_dict()
        assert 'x' in d
        assert 'magnetizacion' in d
        assert 'pureza' in d


class TestSimuladorQubitEstadosIniciales:
    """Tests para estados iniciales."""
    
    def setup_method(self):
        """Configuración antes de cada test."""
        self.sim = SimuladorQubit(seed=42)
    
    def test_estado_0(self):
        """Verifica estado |0⟩."""
        e = self.sim.estado_inicial('0')
        assert np.isclose(e.z, 1.0)
        assert np.isclose(e.x, 0.0)
        assert np.isclose(e.y, 0.0)
    
    def test_estado_1(self):
        """Verifica estado |1⟩."""
        e = self.sim.estado_inicial('1')
        assert np.isclose(e.z, -1.0)
    
    def test_estado_plus(self):
        """Verifica estado |+⟩."""
        e = self.sim.estado_inicial('+')
        assert np.isclose(e.x, 1.0)
        assert np.isclose(e.y, 0.0)
    
    def test_estado_menos(self):
        """Verifica estado |-⟩."""
        e = self.sim.estado_inicial('-')
        assert np.isclose(e.x, -1.0)
    
    def test_estado_invalido(self):
        """Verifica que estado inválido genera error."""
        with pytest.raises(ValueError):
            self.sim.estado_inicial('invalido')


class TestEvolucion:
    """Tests para evoluciones."""
    
    def setup_method(self):
        """Configuración."""
        self.sim = SimuladorQubit(seed=42)
    
    def test_evolucion_libre_rotacion(self):
        """Verifica rotación en evolución libre."""
        estado = self.sim.estado_inicial('+')
        estado_evo = self.sim.evolucion_libre(estado, tiempo=np.pi/2, frecuencia_larmor=1.0)
        
        # Debe haber rotado alrededor de z
        assert abs(estado_evo.magnetizacion - estado.magnetizacion) < 1e-10
    
    def test_evolucion_libre_preserva_pureza(self):
        """Verifica que evolución libre preserva pureza."""
        estado = self.sim.estado_inicial('0')
        estado_evo = self.sim.evolucion_libre(estado, tiempo=1.0)
        assert np.isclose(estado_evo.pureza, estado.pureza)
    
    def test_decoherencia_t1_decae_hacia_0(self):
        """Verifica que T1 decae hacia |0⟩."""
        estado = self.sim.estado_inicial('1')
        estado_dec = self.sim.decoherencia_t1(estado, tiempo=10.0, t1=1.0)
        # Debe estar más cerca de |0⟩
        assert estado_dec.z > estado.z
    
    def test_decoherencia_t2_pierde_coherencia(self):
        """Verifica que T2 pierde coherencia transversal."""
        estado = self.sim.estado_inicial('+')
        estado_dec = self.sim.decoherencia_t2(estado, tiempo=5.0, t2=1.0)
        # Componentes x e y deben decrecer
        assert abs(estado_dec.x) < abs(estado.x)
        assert abs(estado_dec.y) < abs(estado.y)
    
    def test_pulso_resonante_pi_flip(self):
        """Verifica que pulso π voltea el estado."""
        estado = self.sim.estado_inicial('0')
        estado_pi = self.sim.pulso_resonante(estado, duracion=np.pi, amplitud=1.0, tipo='x')
        # Debe estar cerca de |1⟩
        assert estado_pi.z < -0.5
    
    def test_pulso_resonante_pi_2(self):
        """Verifica que pulso π/2 crea superposición."""
        estado = self.sim.estado_inicial('0')
        estado_pi2 = self.sim.pulso_resonante(estado, duracion=np.pi/2, amplitud=1.0, tipo='x')
        # Debe estar en superposición
        assert abs(estado_pi2.z) < 1.0
        assert estado_pi2.magnetizacion > 0


class TestExperimentos:
    """Tests para experimentos."""
    
    def setup_method(self):
        """Configuración."""
        self.sim = SimuladorQubit(seed=42)
    
    def test_rabi_shape(self):
        """Verifica forma de oscilaciones Rabi."""
        estado = self.sim.estado_inicial('0')
        duraciones, probs = self.sim.experimento_rabi(estado)
        
        assert len(duraciones) == len(probs)
        assert np.all((probs >= 0) & (probs <= 1))
    
    def test_rabi_comienza_en_cero(self):
        """Verifica que Rabi comienza en 0."""
        estado = self.sim.estado_inicial('0')
        _, probs = self.sim.experimento_rabi(estado)
        assert np.isclose(probs[0], 0.0, atol=1e-10)
    
    def test_rabi_oscila(self):
        """Verifica que Rabi oscila."""
        estado = self.sim.estado_inicial('0')
        _, probs = self.sim.experimento_rabi(estado, duracion_max=4*np.pi, num_puntos=1000)
        
        # Debe tener máximo y mínimo
        assert np.max(probs) > 0.4
        assert np.min(probs) < 0.1
    
    def test_echo_shape(self):
        """Verifica forma del echo."""
        estado = self.sim.estado_inicial('+')
        tiempos, mag_libre, mag_echo = self.sim.experimento_echo(estado)
        
        assert len(tiempos) == len(mag_libre)
        assert len(tiempos) == len(mag_echo)
    
    def test_echo_mejora_coherencia(self):
        """Verifica que echo mejora coherencia."""
        estado = self.sim.estado_inicial('+')
        _, mag_libre, mag_echo = self.sim.experimento_echo(estado, t1=10.0, t2=1.0)
        
        # El echo debe mantener más coherencia que sin echo
        integral_libre = np.trapz(mag_libre)
        integral_echo = np.trapz(mag_echo)
        assert integral_echo > integral_libre


class TestVisualizacion:
    """Tests para visualización."""
    
    def setup_method(self):
        """Configuración."""
        self.sim = SimuladorQubit(seed=42)
    
    def test_visualizar_sin_archivo(self):
        """Verifica que visualización funciona sin guardar."""
        estado = self.sim.estado_inicial('0')
        # No debería lanzar error
        try:
            self.sim.visualizar_esfera_bloch(estado)
            assert True
        except Exception as e:
            pytest.fail(f"Visualización falló: {e}")


class TestResumen:
    """Tests para resumen."""
    
    def setup_method(self):
        """Configuración."""
        self.sim = SimuladorQubit(seed=42)
    
    def test_resumen_completo(self):
        """Verifica que resumen contiene todo."""
        estado = self.sim.estado_inicial('+')
        resumen = self.sim.resumen_simulacion(estado)
        
        assert 'estado_bloch' in resumen
        assert 'propiedades' in resumen
        assert 'magnetizacion' in resumen['propiedades']
        assert 'pureza' in resumen['propiedades']


class TestNormalizacion:
    """Tests para normalización de estados."""
    
    def setup_method(self):
        """Configuración."""
        self.sim = SimuladorQubit(seed=42)
    
    def test_estado_de_bloch_normaliza(self):
        """Verifica normalización de estados."""
        estado = self.sim.estado_de_bloch(x=2, y=2, z=2)
        mag = estado.magnetizacion
        assert mag <= 1.0 + 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
