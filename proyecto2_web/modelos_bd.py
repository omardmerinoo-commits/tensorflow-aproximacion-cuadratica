"""
Módulo de base de datos para la aplicación web de gestión de experimentos.

Proporciona:
- Modelos SQLAlchemy para persistencia
- Operaciones CRUD completas
- Validación de datos
- Exportación a CSV y JSON
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import csv
import json
from io import StringIO, BytesIO
import pandas as pd

db = SQLAlchemy()


class Experimento(db.Model):
    """Modelo para almacenar experimentos."""
    __tablename__ = 'experimentos'
    
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(255), nullable=False, unique=True)
    descripcion = db.Column(db.Text, nullable=True)
    tipo = db.Column(db.String(100), nullable=False)  # tipo de oscilación
    fecha_creacion = db.Column(db.DateTime, default=datetime.utcnow)
    fecha_modificacion = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    estado = db.Column(db.String(50), default='activo')  # activo, completado, cancelado
    
    # Parámetros del sistema
    masa = db.Column(db.Float, nullable=True)
    amortiguamiento = db.Column(db.Float, nullable=True)
    rigidez = db.Column(db.Float, nullable=True)
    posicion_inicial = db.Column(db.Float, nullable=True)
    velocidad_inicial = db.Column(db.Float, nullable=True)
    
    # Parámetros de simulación
    tiempo_maximo = db.Column(db.Float, default=10.0)
    num_puntos = db.Column(db.Integer, default=100)
    ruido_sigma = db.Column(db.Float, default=0.02)
    
    # Relación con puntos de datos
    puntos_datos = db.relationship('PuntoDato', backref='experimento', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Experimento {self.nombre}>'
    
    def to_dict(self):
        """Convierte el experimento a diccionario."""
        return {
            'id': self.id,
            'nombre': self.nombre,
            'descripcion': self.descripcion,
            'tipo': self.tipo,
            'fecha_creacion': self.fecha_creacion.isoformat(),
            'fecha_modificacion': self.fecha_modificacion.isoformat(),
            'estado': self.estado,
            'parametros': {
                'masa': self.masa,
                'amortiguamiento': self.amortiguamiento,
                'rigidez': self.rigidez,
                'posicion_inicial': self.posicion_inicial,
                'velocidad_inicial': self.velocidad_inicial,
            },
            'simulacion': {
                'tiempo_maximo': self.tiempo_maximo,
                'num_puntos': self.num_puntos,
                'ruido_sigma': self.ruido_sigma,
            },
            'num_puntos_datos': len(self.puntos_datos)
        }
    
    def validar(self):
        """Valida los parámetros del experimento."""
        errores = []
        
        if not self.nombre or len(self.nombre.strip()) == 0:
            errores.append('El nombre del experimento es obligatorio')
        
        if self.masa is not None and self.masa <= 0:
            errores.append('La masa debe ser positiva')
        
        if self.amortiguamiento is not None and self.amortiguamiento < 0:
            errores.append('El amortiguamiento no puede ser negativo')
        
        if self.rigidez is not None and self.rigidez <= 0:
            errores.append('La rigidez debe ser positiva')
        
        if self.tiempo_maximo <= 0:
            errores.append('El tiempo máximo debe ser positivo')
        
        if self.num_puntos < 10:
            errores.append('Se necesitan al menos 10 puntos')
        
        return errores


class PuntoDato(db.Model):
    """Modelo para almacenar puntos de datos de experimentos."""
    __tablename__ = 'puntos_datos'
    
    id = db.Column(db.Integer, primary_key=True)
    experimento_id = db.Column(db.Integer, db.ForeignKey('experimentos.id'), nullable=False)
    tiempo = db.Column(db.Float, nullable=False)
    posicion = db.Column(db.Float, nullable=False)
    velocidad = db.Column(db.Float, nullable=True)
    aceleracion = db.Column(db.Float, nullable=True)
    energia = db.Column(db.Float, nullable=True)
    fecha_creacion = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('experimento_id', 'tiempo', name='unique_exp_tiempo'),
    )
    
    def __repr__(self):
        return f'<PuntoDato exp={self.experimento_id} t={self.tiempo}>'
    
    def to_dict(self):
        """Convierte el punto a diccionario."""
        return {
            'id': self.id,
            'experimento_id': self.experimento_id,
            'tiempo': self.tiempo,
            'posicion': self.posicion,
            'velocidad': self.velocidad,
            'aceleracion': self.aceleracion,
            'energia': self.energia,
        }


class ParametroExperimento(db.Model):
    """Modelo para almacenar parámetros adicionales y metadatos."""
    __tablename__ = 'parametros_experimento'
    
    id = db.Column(db.Integer, primary_key=True)
    experimento_id = db.Column(db.Integer, db.ForeignKey('experimentos.id'), nullable=False)
    clave = db.Column(db.String(100), nullable=False)
    valor = db.Column(db.String(500), nullable=False)
    tipo = db.Column(db.String(50), nullable=False)  # string, number, boolean
    fecha_creacion = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ParametroExperimento {self.clave}={self.valor}>'


class ServicioExperimentos:
    """Servicio para operaciones CRUD de experimentos."""
    
    @staticmethod
    def crear_experimento(datos: dict) -> tuple:
        """
        Crea un nuevo experimento.
        
        Args:
            datos: Diccionario con datos del experimento
            
        Returns:
            Tupla (experimento, lista_errores)
        """
        experimento = Experimento(
            nombre=datos.get('nombre'),
            descripcion=datos.get('descripcion'),
            tipo=datos.get('tipo', 'generico'),
            masa=datos.get('masa'),
            amortiguamiento=datos.get('amortiguamiento'),
            rigidez=datos.get('rigidez'),
            posicion_inicial=datos.get('posicion_inicial'),
            velocidad_inicial=datos.get('velocidad_inicial'),
            tiempo_maximo=datos.get('tiempo_maximo', 10.0),
            num_puntos=datos.get('num_puntos', 100),
            ruido_sigma=datos.get('ruido_sigma', 0.02),
        )
        
        # Validar
        errores = experimento.validar()
        if errores:
            return None, errores
        
        try:
            db.session.add(experimento)
            db.session.commit()
            return experimento, []
        except Exception as e:
            db.session.rollback()
            return None, [f"Error en base de datos: {str(e)}"]
    
    @staticmethod
    def obtener_experimento(id: int):
        """Obtiene un experimento por ID."""
        return Experimento.query.get(id)
    
    @staticmethod
    def listar_experimentos(filtro_estado: str = None, filtro_tipo: str = None):
        """Lista todos los experimentos con filtros opcionales."""
        query = Experimento.query
        
        if filtro_estado:
            query = query.filter_by(estado=filtro_estado)
        
        if filtro_tipo:
            query = query.filter_by(tipo=filtro_tipo)
        
        return query.order_by(Experimento.fecha_creacion.desc()).all()
    
    @staticmethod
    def actualizar_experimento(id: int, datos: dict) -> tuple:
        """Actualiza un experimento."""
        experimento = Experimento.query.get(id)
        if not experimento:
            return None, ["Experimento no encontrado"]
        
        # Actualizar campos
        for key, value in datos.items():
            if hasattr(experimento, key) and key != 'id':
                setattr(experimento, key, value)
        
        # Validar
        errores = experimento.validar()
        if errores:
            return None, errores
        
        try:
            db.session.commit()
            return experimento, []
        except Exception as e:
            db.session.rollback()
            return None, [f"Error en base de datos: {str(e)}"]
    
    @staticmethod
    def eliminar_experimento(id: int) -> bool:
        """Elimina un experimento."""
        experimento = Experimento.query.get(id)
        if not experimento:
            return False
        
        try:
            db.session.delete(experimento)
            db.session.commit()
            return True
        except Exception:
            db.session.rollback()
            return False
    
    @staticmethod
    def agregar_punto_datos(experimento_id: int, punto: dict) -> tuple:
        """Agrega un punto de datos a un experimento."""
        experimento = Experimento.query.get(experimento_id)
        if not experimento:
            return None, ["Experimento no encontrado"]
        
        punto_dato = PuntoDato(
            experimento_id=experimento_id,
            tiempo=punto.get('tiempo'),
            posicion=punto.get('posicion'),
            velocidad=punto.get('velocidad'),
            aceleracion=punto.get('aceleracion'),
            energia=punto.get('energia'),
        )
        
        try:
            db.session.add(punto_dato)
            db.session.commit()
            return punto_dato, []
        except Exception as e:
            db.session.rollback()
            return None, [f"Error: {str(e)}"]
    
    @staticmethod
    def obtener_puntos_datos(experimento_id: int, filtro_tiempo_min: float = None, 
                            filtro_tiempo_max: float = None):
        """Obtiene puntos de datos de un experimento con filtros opcionales."""
        query = PuntoDato.query.filter_by(experimento_id=experimento_id)
        
        if filtro_tiempo_min is not None:
            query = query.filter(PuntoDato.tiempo >= filtro_tiempo_min)
        
        if filtro_tiempo_max is not None:
            query = query.filter(PuntoDato.tiempo <= filtro_tiempo_max)
        
        return query.order_by(PuntoDato.tiempo).all()
    
    @staticmethod
    def exportar_csv(experimento_id: int) -> StringIO:
        """Exporta datos de un experimento a CSV."""
        experimento = Experimento.query.get(experimento_id)
        if not experimento:
            return None
        
        puntos = PuntoDato.query.filter_by(experimento_id=experimento_id).all()
        
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'tiempo', 'posicion', 'velocidad', 'aceleracion', 'energia'
        ])
        
        writer.writeheader()
        for punto in puntos:
            writer.writerow({
                'tiempo': punto.tiempo,
                'posicion': punto.posicion,
                'velocidad': punto.velocidad,
                'aceleracion': punto.aceleracion,
                'energia': punto.energia,
            })
        
        output.seek(0)
        return output
    
    @staticmethod
    def exportar_json(experimento_id: int) -> dict:
        """Exporta datos de un experimento a JSON."""
        experimento = Experimento.query.get(experimento_id)
        if not experimento:
            return None
        
        puntos = PuntoDato.query.filter_by(experimento_id=experimento_id).all()
        
        return {
            'experimento': experimento.to_dict(),
            'puntos_datos': [p.to_dict() for p in puntos]
        }
    
    @staticmethod
    def obtener_estadisticas(experimento_id: int) -> dict:
        """Calcula estadísticas de un experimento."""
        puntos = PuntoDato.query.filter_by(experimento_id=experimento_id).all()
        
        if not puntos:
            return {}
        
        posiciones = [p.posicion for p in puntos]
        velocidades = [p.velocidad for p in puntos if p.velocidad is not None]
        
        import numpy as np
        
        return {
            'num_puntos': len(puntos),
            'tiempo_min': min(p.tiempo for p in puntos),
            'tiempo_max': max(p.tiempo for p in puntos),
            'posicion_min': float(np.min(posiciones)),
            'posicion_max': float(np.max(posiciones)),
            'posicion_media': float(np.mean(posiciones)),
            'posicion_std': float(np.std(posiciones)),
            'velocidad_min': float(np.min(velocidades)) if velocidades else None,
            'velocidad_max': float(np.max(velocidades)) if velocidades else None,
            'energia_media': float(np.mean([p.energia for p in puntos if p.energia is not None])) if any(p.energia for p in puntos) else None,
        }
