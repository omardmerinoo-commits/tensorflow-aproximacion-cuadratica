"""
Tests para la aplicación web de experimentos.
"""

import pytest
import json
from pathlib import Path
import sys

# Agregar la ruta del proyecto
sys.path.insert(0, str(Path(__file__).parent))

from app import app, db
from modelos_bd import Experimento, PuntoDato


@pytest.fixture
def cliente():
    """Fixture con cliente de prueba."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


class TestExperimentos:
    """Tests para endpoints de experimentos."""
    
    def test_listar_experimentos_vacio(self, cliente):
        """Verifica listar experimentos vacío."""
        respuesta = cliente.get('/api/experimentos')
        assert respuesta.status_code == 200
        datos = respuesta.get_json()
        assert datos['exito'] is True
        assert datos['total'] == 0
    
    def test_crear_experimento(self, cliente):
        """Verifica creación de experimento."""
        payload = {
            'nombre': 'Test Exp 1',
            'tipo': 'subamortiguado',
            'masa': 1.0,
            'amortiguamiento': 0.5,
            'rigidez': 2.0
        }
        respuesta = cliente.post('/api/experimentos', json=payload)
        assert respuesta.status_code == 201
        datos = respuesta.get_json()
        assert datos['exito'] is True
        assert 'experimento' in datos
        assert datos['experimento']['nombre'] == 'Test Exp 1'
    
    def test_crear_experimento_sin_nombre_falla(self, cliente):
        """Verifica que falla sin nombre."""
        payload = {'tipo': 'generico'}
        respuesta = cliente.post('/api/experimentos', json=payload)
        assert respuesta.status_code == 400
    
    def test_obtener_experimento(self, cliente):
        """Verifica obtener un experimento."""
        # Crear primero
        payload = {'nombre': 'Test 2', 'tipo': 'generico'}
        respuesta = cliente.post('/api/experimentos', json=payload)
        exp_id = respuesta.get_json()['experimento']['id']
        
        # Obtener
        respuesta = cliente.get(f'/api/experimentos/{exp_id}')
        assert respuesta.status_code == 200
        datos = respuesta.get_json()
        assert datos['experimento']['id'] == exp_id
    
    def test_actualizar_experimento(self, cliente):
        """Verifica actualización de experimento."""
        # Crear
        payload = {'nombre': 'Test 3', 'tipo': 'generico'}
        respuesta = cliente.post('/api/experimentos', json=payload)
        exp_id = respuesta.get_json()['experimento']['id']
        
        # Actualizar
        update_payload = {'estado': 'completado'}
        respuesta = cliente.put(f'/api/experimentos/{exp_id}', json=update_payload)
        assert respuesta.status_code == 200
        datos = respuesta.get_json()
        assert datos['experimento']['estado'] == 'completado'
    
    def test_eliminar_experimento(self, cliente):
        """Verifica eliminación de experimento."""
        # Crear
        payload = {'nombre': 'Test 4', 'tipo': 'generico'}
        respuesta = cliente.post('/api/experimentos', json=payload)
        exp_id = respuesta.get_json()['experimento']['id']
        
        # Eliminar
        respuesta = cliente.delete(f'/api/experimentos/{exp_id}')
        assert respuesta.status_code == 200
        
        # Verificar que no existe
        respuesta = cliente.get(f'/api/experimentos/{exp_id}')
        assert respuesta.status_code == 404


class TestPuntosDatos:
    """Tests para endpoints de puntos de datos."""
    
    @pytest.fixture
    def experimento_id(self, cliente):
        """Fixture que crea un experimento."""
        payload = {'nombre': 'Test Datos', 'tipo': 'generico'}
        respuesta = cliente.post('/api/experimentos', json=payload)
        return respuesta.get_json()['experimento']['id']
    
    def test_obtener_puntos_vacio(self, cliente, experimento_id):
        """Verifica obtener puntos vacío."""
        respuesta = cliente.get(f'/api/experimentos/{experimento_id}/puntos')
        assert respuesta.status_code == 200
        datos = respuesta.get_json()
        assert datos['total'] == 0
    
    def test_agregar_punto(self, cliente, experimento_id):
        """Verifica agregar un punto."""
        payload = {
            'tiempo': 0.0,
            'posicion': 1.0,
            'velocidad': 0.0,
            'aceleracion': -1.0,
            'energia': 0.5
        }
        respuesta = cliente.post(f'/api/experimentos/{experimento_id}/puntos', json=payload)
        assert respuesta.status_code == 201
        datos = respuesta.get_json()
        assert datos['exito'] is True
    
    def test_agregar_puntos_batch(self, cliente, experimento_id):
        """Verifica agregar múltiples puntos."""
        payload = {
            'puntos': [
                {'tiempo': 0.0, 'posicion': 1.0},
                {'tiempo': 0.1, 'posicion': 0.9},
                {'tiempo': 0.2, 'posicion': 0.8},
            ]
        }
        respuesta = cliente.post(f'/api/experimentos/{experimento_id}/puntos/batch', json=payload)
        assert respuesta.status_code == 200
        datos = respuesta.get_json()
        assert datos['puntos_agregados'] == 3
    
    def test_obtener_puntos_con_datos(self, cliente, experimento_id):
        """Verifica obtener puntos después de agregar."""
        # Agregar
        payload = {'tiempo': 0.0, 'posicion': 1.0}
        cliente.post(f'/api/experimentos/{experimento_id}/puntos', json=payload)
        
        # Obtener
        respuesta = cliente.get(f'/api/experimentos/{experimento_id}/puntos')
        datos = respuesta.get_json()
        assert datos['total'] == 1


class TestEstadisticas:
    """Tests para estadísticas."""
    
    @pytest.fixture
    def experimento_con_datos(self, cliente):
        """Crea un experimento con puntos."""
        # Crear experimento
        payload = {'nombre': 'Test Stats', 'tipo': 'generico'}
        respuesta = cliente.post('/api/experimentos', json=payload)
        exp_id = respuesta.get_json()['experimento']['id']
        
        # Agregar puntos
        puntos = {
            'puntos': [
                {'tiempo': float(i)*0.1, 'posicion': float(i), 'velocidad': 1.0, 'energia': 5.0}
                for i in range(10)
            ]
        }
        cliente.post(f'/api/experimentos/{exp_id}/puntos/batch', json=puntos)
        
        return exp_id
    
    def test_obtener_estadisticas(self, cliente, experimento_con_datos):
        """Verifica obtener estadísticas."""
        respuesta = cliente.get(f'/api/experimentos/{experimento_con_datos}/estadisticas')
        assert respuesta.status_code == 200
        datos = respuesta.get_json()
        assert datos['exito'] is True
        assert 'estadisticas' in datos
        stats = datos['estadisticas']
        assert stats['num_puntos'] == 10


class TestExportacion:
    """Tests para exportación."""
    
    @pytest.fixture
    def experimento_id(self, cliente):
        """Crea un experimento con datos."""
        payload = {'nombre': 'Test Export', 'tipo': 'generico'}
        respuesta = cliente.post('/api/experimentos', json=payload)
        exp_id = respuesta.get_json()['experimento']['id']
        
        puntos = {
            'puntos': [
                {'tiempo': 0.0, 'posicion': 1.0},
                {'tiempo': 0.1, 'posicion': 0.9},
            ]
        }
        cliente.post(f'/api/experimentos/{exp_id}/puntos/batch', json=puntos)
        
        return exp_id
    
    def test_exportar_json(self, cliente, experimento_id):
        """Verifica exportar a JSON."""
        respuesta = cliente.get(f'/api/experimentos/{experimento_id}/exportar/json')
        assert respuesta.status_code == 200
        assert 'application/octet-stream' in respuesta.content_type or 'application/json' in respuesta.content_type
    
    def test_exportar_csv(self, cliente, experimento_id):
        """Verifica exportar a CSV."""
        respuesta = cliente.get(f'/api/experimentos/{experimento_id}/exportar/csv')
        assert respuesta.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
