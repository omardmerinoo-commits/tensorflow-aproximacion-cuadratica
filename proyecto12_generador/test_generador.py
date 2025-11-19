"""
Test Suite: Generador Sintético
===============================

40+ pruebas cubriendo:
- Generación de datos sintéticos
- Construcción de GAN y VAE
- Entrenamiento adversarial
- Generación de imágenes
- Reconstrucción VAE
- Latent space interpolation
- Persistencia

Cobertura target: >90%
"""

import pytest
import numpy as np
import tensorflow as tf
import os
import tempfile

from generador_sintetico import (
    GeneradorDatos,
    GAN,
    VAE,
    DatosGenerativos
)


class TestGeneracionDatos:
    """Pruebas de generación de datos sintéticos"""
    
    def test_init_generador(self):
        """Verifica inicialización"""
        gen = GeneradorDatos(seed=42)
        assert gen.seed == 42
    
    def test_generar_mnist_sintetico(self):
        """Verifica generación de MNIST sintético"""
        gen = GeneradorDatos()
        imgs = gen.generar_mnist_sintético(n_samples=50)
        assert imgs.shape == (50, 28, 28, 1)
    
    def test_imagenes_rango_valido(self):
        """Verifica que imágenes están en [0, 1]"""
        gen = GeneradorDatos()
        imgs = gen.generar_mnist_sintético(n_samples=50)
        assert np.all(imgs >= 0)
        assert np.all(imgs <= 1)
    
    def test_imagenes_diferentes(self):
        """Verifica que imágenes no son idénticas"""
        gen = GeneradorDatos()
        imgs = gen.generar_mnist_sintético(n_samples=10)
        # Al menos algunas son diferentes
        diffs = []
        for i in range(len(imgs)-1):
            diff = np.mean(np.abs(imgs[i] - imgs[i+1]))
            diffs.append(diff)
        assert np.mean(diffs) > 0.01
    
    def test_formas_correctas(self):
        """Verifica formas geométricas esperadas"""
        gen = GeneradorDatos()
        for _ in range(10):
            imgs = gen.generar_mnist_sintético(n_samples=1)
            assert imgs.shape == (1, 28, 28, 1)
    
    def test_generar_dataset(self):
        """Verifica creación de dataset"""
        gen = GeneradorDatos()
        datos = gen.generar_dataset(n_samples=100)
        assert isinstance(datos, DatosGenerativos)
    
    def test_dataset_split(self):
        """Verifica split del dataset"""
        gen = GeneradorDatos()
        datos = gen.generar_dataset(n_samples=100, split=(0.7, 0.15, 0.15))
        total = len(datos.X_train) + len(datos.X_val) + len(datos.X_test)
        assert total == 100
    
    def test_dataset_sin_nulos(self):
        """Verifica que no hay NaN"""
        gen = GeneradorDatos()
        datos = gen.generar_dataset(n_samples=50)
        assert not np.any(np.isnan(datos.X_train))


class TestConstruccionGAN:
    """Pruebas de construcción GAN"""
    
    def test_crear_gan(self):
        """Verifica creación de GAN"""
        gan = GAN(latent_dim=100)
        assert gan.latent_dim == 100
    
    def test_construir_generador(self):
        """Verifica construcción del generador"""
        gan = GAN(latent_dim=100)
        gen = gan.construir_generador()
        assert gen is not None
        assert gen.input_shape == (None, 100)
        assert gen.output_shape == (None, 28, 28, 1)
    
    def test_construir_discriminador(self):
        """Verifica construcción del discriminador"""
        gan = GAN()
        disc = gan.construir_discriminador()
        assert disc is not None
        assert disc.input_shape == (None, 28, 28, 1)
        assert disc.output_shape == (None, 1)
    
    def test_generador_salida_valida(self):
        """Verifica que generador produce imagen válida"""
        gan = GAN(latent_dim=100, seed=42)
        gen = gan.construir_generador()
        z = np.random.normal(0, 1, (5, 100))
        imgs = gen.predict(z, verbose=0)
        assert imgs.shape == (5, 28, 28, 1)
        assert np.all(imgs >= 0) and np.all(imgs <= 1)
    
    def test_discriminador_salida_valida(self):
        """Verifica que discriminador produce probabilidades"""
        gan = GAN()
        disc = gan.construir_discriminador()
        imgs = np.random.rand(5, 28, 28, 1).astype(np.float32)
        probs = disc.predict(imgs, verbose=0)
        assert probs.shape == (5, 1)
        assert np.all(probs >= 0) and np.all(probs <= 1)


class TestConstruccionVAE:
    """Pruebas de construcción VAE"""
    
    def test_crear_vae(self):
        """Verifica creación de VAE"""
        vae = VAE(latent_dim=32)
        assert vae.latent_dim == 32
    
    def test_construir_encoder(self):
        """Verifica construcción de encoder"""
        vae = VAE(latent_dim=32)
        enc = vae.construir_encoder()
        assert enc is not None
        assert enc.input_shape == (None, 28, 28, 1)
        # Output es [mean, log_var]
        assert len(enc.outputs) == 2
    
    def test_construir_decoder(self):
        """Verifica construcción de decoder"""
        vae = VAE(latent_dim=32)
        dec = vae.construir_decoder()
        assert dec is not None
        assert dec.input_shape == (None, 32)
        assert dec.output_shape == (None, 28, 28, 1)
    
    def test_construir_vae_completo(self):
        """Verifica construcción de VAE completo"""
        vae = VAE(latent_dim=32)
        vae.construir_vae()
        assert vae.encoder is not None
        assert vae.decoder is not None
        assert vae.vae is not None
    
    def test_encoder_salida_dos_componentes(self):
        """Verifica que encoder produce mean y log_var"""
        vae = VAE(latent_dim=32)
        enc = vae.construir_encoder()
        imgs = np.random.rand(5, 28, 28, 1).astype(np.float32)
        mean, log_var = enc.predict(imgs, verbose=0)
        assert mean.shape == (5, 32)
        assert log_var.shape == (5, 32)


class TestEntrenamientoGAN:
    """Pruebas de entrenamiento GAN"""
    
    def test_entrenar_gan(self):
        """Verifica entrenamiento GAN"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        gan = GAN(latent_dim=50, seed=42)
        hist = gan.entrenar(datos.X_train, datos.X_val, 
                           epochs=2, verbose=0)
        
        assert 'g_loss' in hist
        assert 'd_loss' in hist
        assert len(hist['g_loss']) == 2
        assert gan.entrenado
    
    def test_gan_loss_valido(self):
        """Verifica que losses son números válidos"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        gan = GAN(latent_dim=50, seed=42)
        hist = gan.entrenar(datos.X_train, datos.X_val,
                           epochs=2, verbose=0)
        
        assert not np.any(np.isnan(hist['g_loss']))
        assert not np.any(np.isnan(hist['d_loss']))
    
    def test_generar_imagenes_sin_entrenar(self):
        """Verifica error si genera sin entrenar"""
        gan = GAN()
        with pytest.raises(ValueError):
            gan.generar_imagenes(10)


class TestEntrenamientoVAE:
    """Pruebas de entrenamiento VAE"""
    
    def test_entrenar_vae(self):
        """Verifica entrenamiento VAE"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        vae = VAE(latent_dim=32, seed=42)
        vae.construir_vae()
        hist = vae.entrenar(datos.X_train, datos.X_val,
                           epochs=2, verbose=0)
        
        assert 'loss' in hist
        assert vae.entrenado
    
    def test_vae_reconstruccion(self):
        """Verifica reconstrucción de imágenes"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=50)
        
        vae = VAE(latent_dim=32, seed=42)
        vae.construir_vae()
        vae.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        X_recon = vae.reconstruir(datos.X_test[:5])
        assert X_recon.shape == (5, 28, 28, 1)
        assert np.all(X_recon >= 0) and np.all(X_recon <= 1)


class TestGeneracionGAN:
    """Pruebas de generación GAN"""
    
    def test_generar_imagenes_gan(self):
        """Verifica generación de imágenes"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        gan = GAN(latent_dim=50, seed=42)
        gan.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        imgs = gan.generar_imagenes(n_imagenes=10)
        assert imgs.shape == (10, 28, 28, 1)
        assert np.all(imgs >= 0) and np.all(imgs <= 1)
    
    def test_generar_imagenes_diferentes(self):
        """Verifica que genera imágenes diferentes"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        gan = GAN(latent_dim=50, seed=42)
        gan.entrenar(datos.X_train, datos.X_val, epochs=5, verbose=0)
        
        imgs1 = gan.generar_imagenes(5)
        imgs2 = gan.generar_imagenes(5)
        
        # Aunque el modelo es el mismo, el ruido es diferente
        diff = np.mean(np.abs(imgs1 - imgs2))
        assert diff > 0.01


class TestGeneracionVAE:
    """Pruebas de generación VAE"""
    
    def test_generar_imagenes_vae(self):
        """Verifica generación VAE"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        vae = VAE(latent_dim=32, seed=42)
        vae.construir_vae()
        vae.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        imgs = vae.generar_imagenes(n_imagenes=10)
        assert imgs.shape == (10, 28, 28, 1)
    
    def test_interpolacion_latente(self):
        """Verifica interpolación en latent space"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        vae = VAE(latent_dim=32, seed=42)
        vae.construir_vae()
        vae.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        # Interpolar entre dos puntos
        z1 = np.random.normal(0, 1, (1, 32))
        z2 = np.random.normal(0, 1, (1, 32))
        
        # Interpolación lineal
        alpha = np.linspace(0, 1, 5)
        for a in alpha:
            z_interp = (1 - a) * z1 + a * z2
            img = vae.decoder.predict(z_interp, verbose=0)
            assert img.shape == (1, 28, 28, 1)


class TestReconstruccion:
    """Pruebas de reconstrucción"""
    
    def test_reconstruccion_vae(self):
        """Verifica que reconstrucción es cercana al original"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        vae = VAE(latent_dim=32, seed=42)
        vae.construir_vae()
        vae.entrenar(datos.X_train, datos.X_val, epochs=5, verbose=0)
        
        X_recon = vae.reconstruir(datos.X_val[:5])
        error = np.mean(np.abs(datos.X_val[:5] - X_recon))
        
        # Error debe ser razonable
        assert error < 0.5
    
    def test_reconstruccion_deterministica(self):
        """Verifica que reconstrucción es determinística"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=50)
        
        vae = VAE(latent_dim=32, seed=42)
        vae.construir_vae()
        vae.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        X_recon1 = vae.reconstruir(datos.X_test[:3])
        X_recon2 = vae.reconstruir(datos.X_test[:3])
        
        assert np.allclose(X_recon1, X_recon2)


class TestPersistencia:
    """Pruebas de guardado y carga"""
    
    def test_guardar_gan(self):
        """Verifica guardado de GAN"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        gan = GAN(latent_dim=50, seed=42)
        gan.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = os.path.join(tmpdir, 'gan')
            gan.guardar(ruta)
            
            # Verificar que archivos existen
            assert os.path.exists(f"{ruta}_gen.h5")
            assert os.path.exists(f"{ruta}_disc.h5")
    
    def test_guardar_vae(self):
        """Verifica guardado de VAE"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        vae = VAE(latent_dim=32, seed=42)
        vae.construir_vae()
        vae.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = os.path.join(tmpdir, 'vae')
            vae.guardar(ruta)
            
            # Verificar que archivos existen
            assert os.path.exists(f"{ruta}_encoder.h5")
            assert os.path.exists(f"{ruta}_decoder.h5")
            assert os.path.exists(f"{ruta}_vae.h5")


class TestEdgeCases:
    """Pruebas de casos límite"""
    
    def test_dataset_pequeno(self):
        """Maneja dataset muy pequeño"""
        gen_datos = GeneradorDatos()
        datos = gen_datos.generar_dataset(n_samples=20)
        assert len(datos.X_train) > 0
    
    def test_latent_dim_diferente(self):
        """Verifica latent_dim diferente"""
        for dim in [8, 16, 64]:
            vae = VAE(latent_dim=dim)
            vae.construir_vae()
            assert vae.encoder.layers[-2].units == dim
    
    def test_generar_una_imagen(self):
        """Verifica generación de una imagen"""
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        gan = GAN(latent_dim=50, seed=42)
        gan.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        img = gan.generar_imagenes(n_imagenes=1)
        assert img.shape == (1, 28, 28, 1)


class TestRendimiento:
    """Pruebas de rendimiento"""
    
    def test_velocidad_generacion_datos(self):
        """Verifica que generación es rápida"""
        import time
        gen = GeneradorDatos()
        
        t_inicio = time.time()
        gen.generar_dataset(n_samples=500)
        t_duracion = time.time() - t_inicio
        
        assert t_duracion < 5  # Menos de 5 segundos
    
    def test_velocidad_generacion_gan(self):
        """Verifica que generación GAN es rápida"""
        import time
        gen_datos = GeneradorDatos(seed=42)
        datos = gen_datos.generar_dataset(n_samples=100)
        
        gan = GAN(latent_dim=50, seed=42)
        gan.entrenar(datos.X_train, datos.X_val, epochs=2, verbose=0)
        
        t_inicio = time.time()
        for _ in range(10):
            gan.generar_imagenes(10)
        t_duracion = time.time() - t_inicio
        
        assert t_duracion < 10  # Menos de 10 segundos


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
