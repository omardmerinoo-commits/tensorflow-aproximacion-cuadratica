"""
Cliente CLI para interactuar con la API de experimentos.

Proporciona utilidades de línea de comandos para:
- Crear/listar experimentos
- Generar datos
- Exportar datos
- Consultar estadísticas
"""

import requests
import json
import click
import tabulate
from typing import Optional


class ClienteAPI:
    """Cliente para interactuar con la API."""
    
    def __init__(self, url_base: str = 'http://localhost:5000'):
        """Inicializa el cliente."""
        self.url_base = url_base
    
    def _hacer_peticion(self, metodo: str, endpoint: str, datos: Optional[dict] = None):
        """Realiza una petición HTTP."""
        url = f"{self.url_base}{endpoint}"
        
        try:
            if metodo == 'GET':
                response = requests.get(url)
            elif metodo == 'POST':
                response = requests.post(url, json=datos)
            elif metodo == 'PUT':
                response = requests.put(url, json=datos)
            elif metodo == 'DELETE':
                response = requests.delete(url)
            else:
                raise ValueError(f"Método {metodo} no soportado")
            
            return response.json()
        except requests.exceptions.ConnectionError:
            return {'exito': False, 'error': 'No se puede conectar al servidor'}
        except Exception as e:
            return {'exito': False, 'error': str(e)}
    
    def listar_experimentos(self):
        """Lista todos los experimentos."""
        return self._hacer_peticion('GET', '/api/experimentos')
    
    def crear_experimento(self, nombre: str, tipo: str, **kwargs):
        """Crea un nuevo experimento."""
        datos = {'nombre': nombre, 'tipo': tipo, **kwargs}
        return self._hacer_peticion('POST', '/api/experimentos', datos)
    
    def obtener_experimento(self, id: int):
        """Obtiene un experimento."""
        return self._hacer_peticion('GET', f'/api/experimentos/{id}')
    
    def actualizar_experimento(self, id: int, **kwargs):
        """Actualiza un experimento."""
        return self._hacer_peticion('PUT', f'/api/experimentos/{id}', kwargs)
    
    def eliminar_experimento(self, id: int):
        """Elimina un experimento."""
        return self._hacer_peticion('DELETE', f'/api/experimentos/{id}')
    
    def generar_datos(self, id: int):
        """Genera datos para un experimento."""
        return self._hacer_peticion('POST', f'/api/experimentos/{id}/generar', {})
    
    def obtener_puntos(self, id: int):
        """Obtiene puntos de datos."""
        return self._hacer_peticion('GET', f'/api/experimentos/{id}/puntos')
    
    def obtener_estadisticas(self, id: int):
        """Obtiene estadísticas."""
        return self._hacer_peticion('GET', f'/api/experimentos/{id}/estadisticas')
    
    def exportar_csv(self, id: int):
        """Exporta a CSV."""
        url = f"{self.url_base}/api/experimentos/{id}/exportar/csv"
        try:
            response = requests.get(url)
            return response
        except Exception as e:
            return {'exito': False, 'error': str(e)}
    
    def exportar_json(self, id: int):
        """Exporta a JSON."""
        url = f"{self.url_base}/api/experimentos/{id}/exportar/json"
        try:
            response = requests.get(url)
            return response
        except Exception as e:
            return {'exito': False, 'error': str(e)}


@click.group()
def cli():
    """CLI para gestión de experimentos."""
    pass


cliente = ClienteAPI()


@cli.command()
def listar():
    """Lista todos los experimentos."""
    resultado = cliente.listar_experimentos()
    
    if not resultado.get('exito'):
        click.echo(f"Error: {resultado.get('error')}", err=True)
        return
    
    experimentos = resultado.get('experimentos', [])
    
    if not experimentos:
        click.echo("No hay experimentos.")
        return
    
    # Preparar datos para tabla
    datos_tabla = []
    for exp in experimentos:
        datos_tabla.append([
            exp['id'],
            exp['nombre'],
            exp['tipo'],
            exp['estado'],
            exp['fecha_creacion'][:10],
            exp['num_puntos_datos']
        ])
    
    headers = ['ID', 'Nombre', 'Tipo', 'Estado', 'Fecha', 'Puntos']
    click.echo(tabulate.tabulate(datos_tabla, headers=headers, tablefmt='grid'))
    click.echo(f"\nTotal: {resultado['total']} experimentos")


@cli.command()
@click.option('--nombre', prompt='Nombre del experimento', help='Nombre único')
@click.option('--tipo', default='generico', help='Tipo de oscilación')
@click.option('--masa', type=float, default=1.0, help='Masa (kg)')
@click.option('--amortiguamiento', type=float, default=0.5, help='Coeficiente de amortiguamiento')
@click.option('--rigidez', type=float, default=1.0, help='Constante de rigidez')
def crear(nombre, tipo, masa, amortiguamiento, rigidez):
    """Crea un nuevo experimento."""
    resultado = cliente.crear_experimento(
        nombre=nombre,
        tipo=tipo,
        masa=masa,
        amortiguamiento=amortiguamiento,
        rigidez=rigidez
    )
    
    if resultado.get('exito'):
        exp = resultado['experimento']
        click.echo(f"✓ Experimento creado (ID: {exp['id']})")
        click.echo(json.dumps(exp, indent=2, ensure_ascii=False))
    else:
        click.echo(f"Error: {resultado.get('error')}", err=True)


@cli.command()
@click.argument('id', type=int)
def obtener(id):
    """Obtiene detalles de un experimento."""
    resultado = cliente.obtener_experimento(id)
    
    if resultado.get('exito'):
        click.echo(json.dumps(resultado['experimento'], indent=2, ensure_ascii=False))
    else:
        click.echo(f"Error: {resultado.get('error')}", err=True)


@cli.command()
@click.argument('id', type=int)
def generar(id):
    """Genera datos para un experimento."""
    with click.progressbar(length=100, label='Generando datos') as bar:
        resultado = cliente.generar_datos(id)
        bar.update(100)
    
    if resultado.get('exito'):
        click.echo(f"✓ {resultado.get('mensaje')}")
        click.echo(f"Puntos generados: {resultado.get('puntos_generados')}")
    else:
        click.echo(f"Error: {resultado.get('error')}", err=True)


@cli.command()
@click.argument('id', type=int)
def estadisticas(id):
    """Obtiene estadísticas de un experimento."""
    resultado = cliente.obtener_estadisticas(id)
    
    if resultado.get('exito'):
        stats = resultado['estadisticas']
        for clave, valor in stats.items():
            if isinstance(valor, float):
                click.echo(f"  {clave:.<30} {valor:.6f}")
            else:
                click.echo(f"  {clave:.<30} {valor}")
    else:
        click.echo(f"Error: {resultado.get('error')}", err=True)


@cli.command()
@click.argument('id', type=int)
@click.option('--formato', type=click.Choice(['csv', 'json']), default='csv')
@click.option('--salida', type=click.Path(), help='Ruta de salida')
def exportar(id, formato, salida):
    """Exporta datos de un experimento."""
    click.echo(f"Exportando experimento {id} a {formato.upper()}...")
    
    if formato == 'csv':
        response = cliente.exportar_csv(id)
    else:
        response = cliente.exportar_json(id)
    
    if hasattr(response, 'status_code') and response.status_code == 200:
        filename = salida or f"experimento_{id}.{formato}"
        with open(filename, 'wb') as f:
            f.write(response.content)
        click.echo(f"✓ Exportado a {filename}")
    else:
        click.echo(f"Error: {response.get('error', 'Error desconocido')}", err=True)


@cli.command()
@click.argument('id', type=int)
@click.confirmation_option(prompt='¿Estás seguro?')
def eliminar(id):
    """Elimina un experimento."""
    resultado = cliente.eliminar_experimento(id)
    
    if resultado.get('exito'):
        click.echo(f"✓ Experimento {id} eliminado")
    else:
        click.echo(f"Error: {resultado.get('error')}", err=True)


if __name__ == '__main__':
    cli()
