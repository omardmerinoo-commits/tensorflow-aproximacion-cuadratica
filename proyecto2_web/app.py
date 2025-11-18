"""
Aplicación web Flask para gestión de experimentos.

API REST con endpoints para:
- CRUD de experimentos
- Generación de datos
- Exportación (CSV, JSON)
- Estadísticas
- Visualizaciones
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import json

from modelos_bd import db, Experimento, PuntoDato, ServicioExperimentos

# ==================== CONFIGURACIÓN ====================

class Config:
    """Configuración de la aplicación."""
    SQLALCHEMY_DATABASE_URI = 'sqlite:///experimentos.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'clave-super-secreta-desarrollo'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max


app = Flask(__name__)
app.config.from_object(Config)

# Inicializar extensiones
db.init_app(app)
CORS(app)

# Crear carpetas necesarias
Path('uploads').mkdir(exist_ok=True)
Path('exports').mkdir(exist_ok=True)


# ==================== INICIALIZACIÓN ====================

@app.before_first_request
def crear_tablas():
    """Crea las tablas de la base de datos."""
    db.create_all()


# ==================== RUTAS API: EXPERIMENTOS ====================

@app.route('/api/experimentos', methods=['GET'])
def listar_experimentos():
    """Lista todos los experimentos con filtros opcionales."""
    estado = request.args.get('estado')
    tipo = request.args.get('tipo')
    
    experimentos = ServicioExperimentos.listar_experimentos(
        filtro_estado=estado, 
        filtro_tipo=tipo
    )
    
    return jsonify({
        'exito': True,
        'total': len(experimentos),
        'experimentos': [e.to_dict() for e in experimentos]
    })


@app.route('/api/experimentos/<int:id>', methods=['GET'])
def obtener_experimento(id):
    """Obtiene un experimento específico."""
    experimento = ServicioExperimentos.obtener_experimento(id)
    
    if not experimento:
        return jsonify({'exito': False, 'error': 'Experimento no encontrado'}), 404
    
    return jsonify({
        'exito': True,
        'experimento': experimento.to_dict()
    })


@app.route('/api/experimentos', methods=['POST'])
def crear_experimento():
    """Crea un nuevo experimento."""
    datos = request.get_json()
    
    experimento, errores = ServicioExperimentos.crear_experimento(datos)
    
    if errores:
        return jsonify({
            'exito': False,
            'errores': errores
        }), 400
    
    return jsonify({
        'exito': True,
        'mensaje': 'Experimento creado exitosamente',
        'experimento': experimento.to_dict()
    }), 201


@app.route('/api/experimentos/<int:id>', methods=['PUT'])
def actualizar_experimento(id):
    """Actualiza un experimento."""
    datos = request.get_json()
    
    experimento, errores = ServicioExperimentos.actualizar_experimento(id, datos)
    
    if errores:
        return jsonify({
            'exito': False,
            'errores': errores
        }), 400
    
    return jsonify({
        'exito': True,
        'mensaje': 'Experimento actualizado',
        'experimento': experimento.to_dict()
    })


@app.route('/api/experimentos/<int:id>', methods=['DELETE'])
def eliminar_experimento(id):
    """Elimina un experimento."""
    exito = ServicioExperimentos.eliminar_experimento(id)
    
    if not exito:
        return jsonify({
            'exito': False,
            'error': 'Experimento no encontrado'
        }), 404
    
    return jsonify({
        'exito': True,
        'mensaje': 'Experimento eliminado'
    })


# ==================== RUTAS API: DATOS ====================

@app.route('/api/experimentos/<int:id>/puntos', methods=['GET'])
def obtener_puntos_datos(id):
    """Obtiene puntos de datos de un experimento."""
    tiempo_min = request.args.get('tiempo_min', type=float)
    tiempo_max = request.args.get('tiempo_max', type=float)
    
    puntos = ServicioExperimentos.obtener_puntos_datos(
        id, 
        filtro_tiempo_min=tiempo_min,
        filtro_tiempo_max=tiempo_max
    )
    
    if not puntos and not ServicioExperimentos.obtener_experimento(id):
        return jsonify({'exito': False, 'error': 'Experimento no encontrado'}), 404
    
    return jsonify({
        'exito': True,
        'total': len(puntos),
        'puntos': [p.to_dict() for p in puntos]
    })


@app.route('/api/experimentos/<int:id>/puntos', methods=['POST'])
def agregar_punto_dato(id):
    """Agrega un punto de datos a un experimento."""
    datos = request.get_json()
    
    punto, errores = ServicioExperimentos.agregar_punto_datos(id, datos)
    
    if errores:
        return jsonify({
            'exito': False,
            'errores': errores
        }), 400
    
    return jsonify({
        'exito': True,
        'mensaje': 'Punto agregado',
        'punto': punto.to_dict()
    }), 201


@app.route('/api/experimentos/<int:id>/puntos/batch', methods=['POST'])
def agregar_puntos_batch(id):
    """Agrega múltiples puntos de datos de una sola vez."""
    datos = request.get_json()
    puntos_list = datos.get('puntos', [])
    
    if not puntos_list:
        return jsonify({
            'exito': False,
            'error': 'No hay puntos para agregar'
        }), 400
    
    contador = 0
    errores = []
    
    for punto in puntos_list:
        punto_obj, errs = ServicioExperimentos.agregar_punto_datos(id, punto)
        if errs:
            errores.extend(errs)
        else:
            contador += 1
    
    return jsonify({
        'exito': len(errores) == 0,
        'puntos_agregados': contador,
        'errores': errores
    })


# ==================== RUTAS API: GENERACIÓN DE DATOS ====================

@app.route('/api/experimentos/<int:id>/generar', methods=['POST'])
def generar_datos_experimento(id):
    """
    Genera datos sintéticos para un experimento basado en su configuración.
    """
    experimento = ServicioExperimentos.obtener_experimento(id)
    
    if not experimento:
        return jsonify({'exito': False, 'error': 'Experimento no encontrado'}), 404
    
    from oscilaciones_amortiguadas import OscilacionesAmortiguadas
    
    try:
        modelo = OscilacionesAmortiguadas(seed=42)
        
        # Generar datos
        t = np.linspace(0, experimento.tiempo_maximo, experimento.num_puntos)
        x = modelo.solucion_analitica(
            t,
            m=experimento.masa or 1.0,
            c=experimento.amortiguamiento or 1.0,
            k=experimento.rigidez or 1.0,
            x0=experimento.posicion_inicial or 1.0,
            v0=experimento.velocidad_inicial or 0.0
        )
        
        # Agregar ruido
        x_ruidoso = x + np.random.normal(0, experimento.ruido_sigma, len(t))
        
        # Calcular derivadas (velocidad, aceleración)
        v = np.gradient(x_ruidoso, t)
        a = np.gradient(v, t)
        E = 0.5 * (experimento.masa or 1.0) * v**2 + 0.5 * (experimento.rigidez or 1.0) * x_ruidoso**2
        
        # Agregar puntos a la base de datos
        puntos_agregados = 0
        for ti, xi, vi, ai, Ei in zip(t, x_ruidoso, v, a, E):
            punto_data = {
                'tiempo': float(ti),
                'posicion': float(xi),
                'velocidad': float(vi),
                'aceleracion': float(ai),
                'energia': float(Ei)
            }
            punto, _ = ServicioExperimentos.agregar_punto_datos(id, punto_data)
            if punto:
                puntos_agregados += 1
        
        # Actualizar estado del experimento
        experimento.estado = 'completado'
        db.session.commit()
        
        return jsonify({
            'exito': True,
            'mensaje': f'Generados {puntos_agregados} puntos de datos',
            'puntos_generados': puntos_agregados,
            'experimento': experimento.to_dict()
        })
    
    except Exception as e:
        return jsonify({
            'exito': False,
            'error': f'Error generando datos: {str(e)}'
        }), 500


# ==================== RUTAS API: ESTADÍSTICAS ====================

@app.route('/api/experimentos/<int:id>/estadisticas', methods=['GET'])
def obtener_estadisticas(id):
    """Obtiene estadísticas de un experimento."""
    experimento = ServicioExperimentos.obtener_experimento(id)
    
    if not experimento:
        return jsonify({'exito': False, 'error': 'Experimento no encontrado'}), 404
    
    stats = ServicioExperimentos.obtener_estadisticas(id)
    
    return jsonify({
        'exito': True,
        'experimento_id': id,
        'estadisticas': stats
    })


# ==================== RUTAS API: EXPORTACIÓN ====================

@app.route('/api/experimentos/<int:id>/exportar/csv', methods=['GET'])
def exportar_csv(id):
    """Exporta datos de un experimento a CSV."""
    experimento = ServicioExperimentos.obtener_experimento(id)
    
    if not experimento:
        return jsonify({'exito': False, 'error': 'Experimento no encontrado'}), 404
    
    csv_file = ServicioExperimentos.exportar_csv(id)
    
    if not csv_file:
        return jsonify({'exito': False, 'error': 'No hay datos para exportar'}), 400
    
    # Guardar archivo temporalmente
    filename = f"experimento_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = f"exports/{filename}"
    
    with open(filepath, 'w') as f:
        f.write(csv_file.getvalue())
    
    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route('/api/experimentos/<int:id>/exportar/json', methods=['GET'])
def exportar_json(id):
    """Exporta datos de un experimento a JSON."""
    experimento = ServicioExperimentos.obtener_experimento(id)
    
    if not experimento:
        return jsonify({'exito': False, 'error': 'Experimento no encontrado'}), 404
    
    data = ServicioExperimentos.exportar_json(id)
    
    if not data:
        return jsonify({'exito': False, 'error': 'No hay datos para exportar'}), 400
    
    filename = f"experimento_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = f"exports/{filename}"
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    return send_file(filepath, as_attachment=True, download_name=filename)


# ==================== RUTAS WEB ====================

@app.route('/')
def index():
    """Página principal."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gestor de Experimentos</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: Arial, sans-serif; background: #f5f5f5; color: #333; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; margin-bottom: 20px; }
            .card { background: white; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            .btn:hover { background: #2980b9; }
            .info { background: #e8f4f8; border-left: 4px solid #3498db; padding: 15px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Gestor de Experimentos</h1>
            <div class="info">
                <p><strong>Documentación API:</strong> Ver <code>/api/docs</code></p>
                <p><strong>Estado:</strong> ✓ Servidor ejecutándose correctamente</p>
            </div>
            <div class="card">
                <h2>Endpoints Disponibles</h2>
                <ul style="line-height: 2;">
                    <li><code>GET /api/experimentos</code> - Listar experimentos</li>
                    <li><code>POST /api/experimentos</code> - Crear experimento</li>
                    <li><code>GET /api/experimentos/&lt;id&gt;</code> - Obtener experimento</li>
                    <li><code>PUT /api/experimentos/&lt;id&gt;</code> - Actualizar experimento</li>
                    <li><code>DELETE /api/experimentos/&lt;id&gt;</code> - Eliminar experimento</li>
                    <li><code>POST /api/experimentos/&lt;id&gt;/generar</code> - Generar datos</li>
                    <li><code>GET /api/experimentos/&lt;id&gt;/estadisticas</code> - Obtener estadísticas</li>
                    <li><code>GET /api/experimentos/&lt;id&gt;/exportar/csv</code> - Exportar CSV</li>
                    <li><code>GET /api/experimentos/&lt;id&gt;/exportar/json</code> - Exportar JSON</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''


# ==================== MANEJO DE ERRORES ====================

@app.errorhandler(404)
def no_encontrado(e):
    """Maneja rutas no encontradas."""
    return jsonify({'exito': False, 'error': 'Recurso no encontrado'}), 404


@app.errorhandler(500)
def error_interno(e):
    """Maneja errores internos."""
    return jsonify({'exito': False, 'error': 'Error interno del servidor'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
