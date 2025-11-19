"""
Run Training: Generador Sintético
================================

Demostración completa de GAN + VAE.

Pasos:
1. Generar datos sintéticos
2. Entrenar GAN
3. Entrenar VAE
4. Comparar generaciones
5. Interpolación latente
6. Reconstrucción VAE
"""

import numpy as np
from generador_sintetico import (
    GeneradorDatos,
    GAN,
    VAE
)
import time


def print_section(titulo):
    print("\n" + "="*70)
    print(titulo)
    print("="*70)


def step1_generar_datos():
    """Paso 1: Generar datos"""
    print_section("[1] GENERACIÓN DE DATOS SINTÉTICOS")
    
    gen = GeneradorDatos(seed=42)
    datos = gen.generar_dataset(n_samples=500, split=(0.7, 0.15, 0.15))
    
    print(f"\n✓ {datos.info()}")
    print(f"  - Formas: Círculos, Cuadrados, Triángulos")
    print(f"  - Tamaño imagen: 28×28 (784 píxeles)")
    print(f"  - Rango: [0, 1] (escala de grises)")
    
    return datos


def step2_entrenar_gan(datos):
    """Paso 2: Entrenar GAN"""
    print_section("[2] ENTRENAMIENTO GAN")
    
    print("\n  Arquitectura:")
    print("  - Generador: Dense → Conv2DTranspose × 2 → Output")
    print("  - Discriminador: Conv2D × 3 → GlobalPool → Dense(1)")
    
    gan = GAN(latent_dim=100, seed=42)
    
    print("\n  Entrenando (50 épocas)...")
    t_inicio = time.time()
    
    hist = gan.entrenar(
        datos.X_train, datos.X_val,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    t_duracion = time.time() - t_inicio
    
    print(f"✓ Entrenamiento completado en {t_duracion:.2f}s")
    print(f"\n  Histórico de pérdidas:")
    print(f"  - Epoch 1:  G={hist['g_loss'][0]:.4f}, D={hist['d_loss'][0]:.4f}")
    print(f"  - Epoch 25: G={hist['g_loss'][24]:.4f}, D={hist['d_loss'][24]:.4f}")
    print(f"  - Epoch 50: G={hist['g_loss'][-1]:.4f}, D={hist['d_loss'][-1]:.4f}")
    
    return gan, hist


def step3_entrenar_vae(datos):
    """Paso 3: Entrenar VAE"""
    print_section("[3] ENTRENAMIENTO VAE")
    
    print("\n  Arquitectura:")
    print("  - Latent dim: 32")
    print("  - Encoder: Conv2D × 2 + MaxPool → Dense → [mean, log_var]")
    print("  - Decoder: Dense → Conv2DTranspose × 2 → Output")
    
    vae = VAE(latent_dim=32, seed=42)
    vae.construir_vae()
    
    print("\n  Entrenando (30 épocas)...")
    t_inicio = time.time()
    
    hist = vae.entrenar(
        datos.X_train, datos.X_val,
        epochs=30,
        batch_size=32,
        verbose=0
    )
    
    t_duracion = time.time() - t_inicio
    
    print(f"✓ Entrenamiento completado en {t_duracion:.2f}s")
    print(f"\n  Histórico de pérdidas:")
    print(f"  - Epoch 1:  Loss={hist['loss'][0]:.4f}")
    print(f"  - Epoch 15: Loss={hist['loss'][14]:.4f}")
    print(f"  - Epoch 30: Loss={hist['loss'][-1]:.4f}")
    
    return vae, hist


def step4_generacion_gan(gan):
    """Paso 4: Generación GAN"""
    print_section("[4] GENERACIÓN CON GAN")
    
    print("\n  Generando imágenes sintéticas...")
    imgs = gan.generar_imagenes(n_imagenes=10)
    
    print(f"✓ Generadas {imgs.shape[0]} imágenes")
    print(f"  - Shape: {imgs.shape}")
    print(f"  - Rango: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"  - Mean:  {imgs.mean():.3f}")
    print(f"  - Std:   {imgs.std():.3f}")


def step5_generacion_vae(vae):
    """Paso 5: Generación VAE"""
    print_section("[5] GENERACIÓN CON VAE")
    
    print("\n  Generando imágenes desde latent space...")
    imgs = vae.generar_imagenes(n_imagenes=10)
    
    print(f"✓ Generadas {imgs.shape[0]} imágenes")
    print(f"  - Shape: {imgs.shape}")
    print(f"  - Rango: [{imgs.min():.3f}, {imgs.max():.3f}]")


def step6_reconstruccion(vae, datos):
    """Paso 6: Reconstrucción VAE"""
    print_section("[6] RECONSTRUCCIÓN VAE")
    
    print("\n  Reconstruyendo imágenes del test set...")
    X_recon = vae.reconstruir(datos.X_test[:10])
    
    # Errores
    errors = np.abs(datos.X_test[:10] - X_recon)
    error_mean = np.mean(errors)
    error_max = np.max(errors)
    error_min = np.min(errors)
    
    print(f"✓ Reconstrucción completada")
    print(f"\n  Errores de reconstrucción:")
    print(f"  - Media:  {error_mean:.4f}")
    print(f"  - Máximo: {error_max:.4f}")
    print(f"  - Mínimo: {error_min:.4f}")
    print(f"  - Std:    {np.std(errors):.4f}")


def step7_interpolacion(vae):
    """Paso 7: Interpolación en latent space"""
    print_section("[7] INTERPOLACIÓN EN LATENT SPACE")
    
    print("\n  Interpolando entre dos puntos aleatorios...")
    
    z1 = np.random.normal(0, 1, (1, 32))
    z2 = np.random.normal(0, 1, (1, 32))
    
    alpha_values = np.linspace(0, 1, 5)
    
    print(f"\n  Secuencia de interpolación (5 pasos):")
    for i, alpha in enumerate(alpha_values):
        z_interp = (1 - alpha) * z1 + alpha * z2
        img = vae.decoder.predict(z_interp, verbose=0)
        print(f"    Paso {i+1} (α={alpha:.2f}): Imagen generada")
    
    print(f"\n✓ Interpolación demuestra latent space continuo")


def step8_comparacion(gan, vae, datos):
    """Paso 8: Comparación de modelos"""
    print_section("[8] COMPARACIÓN GAN vs VAE")
    
    print("\n┌─────────────────┬───────────────┬───────────────┐")
    print("│ Característica  │ GAN           │ VAE           │")
    print("├─────────────────┼───────────────┼───────────────┤")
    print("│ Enfoque         │ Adversarial   │ Probabilístico│")
    print("│ Loss            │ JS divergence │ ELBO          │")
    print("│ Latent Space    │ Discreto      │ Continuo      │")
    print("│ Interpolación   │ Áspera        │ Suave         │")
    print("│ Velocidad       │ Rápida        │ Moderada      │")
    print("│ Estabilidad     │ Difícil       │ Estable       │")
    print("└─────────────────┴───────────────┴───────────────┘")
    
    # Generar ejemplos
    print(f"\n  Ejemplos de generación:")
    
    gan_imgs = gan.generar_imagenes(5)
    vae_imgs = vae.generar_imagenes(5)
    
    print(f"    GAN: {gan_imgs.shape} | Media={gan_imgs.mean():.3f}")
    print(f"    VAE: {vae_imgs.shape} | Media={vae_imgs.mean():.3f}")
    
    # Reconstrucción VAE
    X_recon = vae.reconstruir(datos.X_test[:5])
    recon_error = np.mean(np.abs(datos.X_test[:5] - X_recon))
    
    print(f"\n  Error reconstrucción VAE: {recon_error:.4f}")


def main():
    """Ejecuta demostración completa"""
    print("\n" + "="*70)
    print("GENERADOR SINTÉTICO CON GAN + VAE")
    print("="*70)
    
    # Paso 1: Datos
    datos = step1_generar_datos()
    
    # Paso 2: GAN
    gan, hist_gan = step2_entrenar_gan(datos)
    
    # Paso 3: VAE
    vae, hist_vae = step3_entrenar_vae(datos)
    
    # Paso 4: Generación GAN
    step4_generacion_gan(gan)
    
    # Paso 5: Generación VAE
    step5_generacion_vae(vae)
    
    # Paso 6: Reconstrucción
    step6_reconstruccion(vae, datos)
    
    # Paso 7: Interpolación
    step7_interpolacion(vae)
    
    # Paso 8: Comparación
    step8_comparacion(gan, vae, datos)
    
    # Resumen
    print_section("RESUMEN")
    print("\n✓ Demostración completada exitosamente")
    print("\n  Componentes probados:")
    print("    ✓ Generación de datos sintéticos")
    print("    ✓ GAN: Generador + Discriminador")
    print("    ✓ VAE: Encoder + Decoder")
    print("    ✓ Entrenamiento adversarial")
    print("    ✓ Generación de imágenes")
    print("    ✓ Reconstrucción")
    print("    ✓ Interpolación en latent space")
    
    print("\n" + "="*70)
    print("FIN DE LA DEMOSTRACIÓN")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
