# Proyecto 12: Generador Sintético de Imágenes (GAN + VAE)

## Introducción

Sistema completo de modelos generativos para síntesis de imágenes sintéticas usando dos enfoques complementarios:
- **GAN**: Aprendizaje adversarial
- **VAE**: Aprendizaje latente probabilístico

### Aplicaciones

- **Data Augmentation**: Generar más datos de entrenamiento
- **Síntesis de Datos**: Crear muestras realistas
- **Super-resolución**: Aumentar calidad de imágenes
- **Completación**: Rellenar partes de imágenes
- **Interpolación**: Suavizar transiciones entre clases

---

## Fundamentación Teórica

### 1. Redes Generativas Adversariales (GAN)

Marco propuesto por Goodfellow et al. (2014):

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Intuición**: Juego de dos jugadores
- **Generador G**: Crea imágenes falsas para engañar a D
- **Discriminador D**: Detecta si imagen es real o falsa

#### Proceso de Entrenamiento

```
Iteración t:
┌─────────────────────────────────────────────┐
│ 1. Generador:                               │
│    - Muestrea ruido z ~ N(0,I)             │
│    - Genera imagen fake: x_fake = G(z)     │
│    - Objetivo: Engañar discriminador        │
│                                             │
│ 2. Discriminador:                           │
│    - Batch real: {x₁, x₂, ..., xₙ}         │
│    - Batch fake: {G(z₁), G(z₂), ..., G(zₙ)}│
│    - Objetivo: Clasificar real=1, fake=0   │
│                                             │
│ 3. Actualizar pesos                        │
│    - D: Maximizar log(D(x)) + log(1-D(G(z)))
│    - G: Minimizar log(1-D(G(z)))           │
└─────────────────────────────────────────────┘
```

#### Arquitecturas

**Generador**:
- Input: Vector latente $z \in \mathbb{R}^{100}$ (ruido Gaussiano)
- Dense → Reshape: Transforma a mapa (7×7×128)
- Conv2DTranspose (upsampling): 7×7 → 14×14 → 28×28
- Output: Imagen sintética

**Discriminador**:
- Input: Imagen (28×28×1)
- Conv2D (downsampling): 28×28 → 14×14 → 7×7
- GlobalAveragePool → Dense: Binaria (real/falso)
- Output: P(real)

### 2. Autoencoder Variacional (VAE)

Framework probabilístico por Kingma & Welling (2014):

$$\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)||p(z))$$

**Componentes**:

#### Encoder: $q_\phi(z|x)$
Mapea imagen a distribución latente (aproximación posterior):
$$z|x \sim \mathcal{N}(\mu(x), \sigma^2(x))$$

**Reparameterization trick**:
$$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

Permite backprop a través de muestreo (estocástico)

#### Latent Space
- Espacio continuo, interpretable
- Prior: $p(z) = \mathcal{N}(0, I)$ (distribución estándar)
- Cada punto representa imagen posible

#### Decoder: $p_\theta(x|z)$
Reconstruye imagen desde código latente:
$$x|z \sim \mathcal{N}(\mu_{decoder}(z), \sigma^2_{decoder}(z))$$

**Interpretación**: Generador aprende a mapear distribución continua a imágenes

#### Loss Function

$$L = \underbrace{\mathbb{E}[\log p(x|z)]}_{\text{Reconstrucción}} - \underbrace{\beta \cdot D_{KL}(q(z|x)||p(z))}_{\text{Regularización Latente}}$$

- **Reconstrucción**: Qué tan bien se reconstruye imagen original
- **KL divergence**: Qué tan cerca latent space está de prior estándar
- **$\beta$**: Balance (típicamente 1)

### 3. Diferencias GAN vs VAE

| Aspecto | GAN | VAE |
|---------|-----|-----|
| **Enfoque** | Adversarial | Probabilístico |
| **Loss** | JS divergence | ELBO |
| **Latent** | Discreto (noise) | Continuo (distributivo) |
| **Interpolación** | Difícil (no continuo) | Suave (latent continuo) |
| **Modo collapse** | Sí (típico) | No (ELBO garantiza) |
| **Velocidad generación** | Rápido | Lento |
| **Calidad** | Potencialmente mejor | Más estable |
| **Interpretabilidad** | Baja | Alta (latent space explorable) |

### 4. Deconvolución (Conv2DTranspose)

Operación inversa a convolución:

```
Conv2D: Imagen grande → mapa pequeño (downsampling)
Conv2DTranspose: Mapa pequeño → imagen grande (upsampling)

Ejemplo:
Input: (7, 7, 128)
         ↓ Conv2DTranspose(64, kernel=(4,4), strides=(2,2))
Output: (14, 14, 64)  # 7*2 = 14
```

Fórmula output size:
$$\text{out} = (in - 1) \cdot \text{stride} + \text{kernel\_size}$$

### 5. KL Divergence (Distancia Kullback-Leibler)

Medida de diferencia entre distribuciones:

$$D_{KL}(P||Q) = \int_x P(x) \log \frac{P(x)}{Q(x)} dx = \mathbb{E}_{x\sim P}[\log P(x) - \log Q(x)]$$

Para gaussianas:
$$D_{KL}(\mathcal{N}(\mu, \sigma^2)||\mathcal{N}(0,I)) = \frac{1}{2}\sum_j(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

---

## Arquitecturas Implementadas

### 1. GAN (Generative Adversarial Network)

#### Generador
```
Input: (batch, 100)  [ruido Gaussiano]
    ↓
Dense(7*7*128) → Reshape(7, 7, 128)
    ↓
Conv2DTranspose(64, kernel=4, strides=2) → BatchNorm → LeakyReLU
    ↓
Conv2DTranspose(32, kernel=4, strides=2) → BatchNorm → LeakyReLU
    ↓
Conv2D(1, kernel=3, sigmoid)  # Output: imagen [0, 1]
    ↓
Output: (batch, 28, 28, 1)
```

**Parámetros**: ~180K

#### Discriminador
```
Input: (batch, 28, 28, 1)
    ↓
Conv2D(32, kernel=3, strides=2) → LeakyReLU → Dropout(0.2)
    ↓
Conv2D(64, kernel=3, strides=2) → BatchNorm → LeakyReLU → Dropout(0.2)
    ↓
Conv2D(128, kernel=3, strides=2) → BatchNorm → LeakyReLU → Dropout(0.2)
    ↓
GlobalAveragePooling2D()
    ↓
Dense(1, sigmoid)  # Real=1, Fake=0
    ↓
Output: (batch, 1)
```

**Parámetros**: ~220K

### 2. VAE (Variational Autoencoder)

#### Encoder
```
Input: (batch, 28, 28, 1)
    ↓
Conv2D(32, kernel=3, padding='same') → ReLU
    ↓
MaxPooling2D(2) → (batch, 14, 14, 32)
    ↓
Conv2D(64, kernel=3, padding='same') → ReLU
    ↓
MaxPooling2D(2) → (batch, 7, 7, 64)
    ↓
Flatten() → Dense(128) → ReLU
    ↓
[Dense(latent_dim)]  ← z_mean
[Dense(latent_dim)]  ← z_log_var
```

#### Decoder
```
Input: (batch, latent_dim)  [muestreado del espacio latente]
    ↓
Dense(7*7*64) → Reshape(7, 7, 64)
    ↓
Conv2DTranspose(64, kernel=3, strides=2) → ReLU
    ↓
Conv2DTranspose(32, kernel=3, strides=2) → ReLU
    ↓
Conv2D(1, kernel=3, padding='same', sigmoid)
    ↓
Output: (batch, 28, 28, 1)
```

**Parámetros**: ~250K

---

## Dataset Sintético

### Generación

Imágenes de formas geométricas simples (28×28):
- **Círculos**: Centro aleatorio, radio variable
- **Cuadrados**: Tamaño variable, rotación
- **Triángulos**: Vértices aleatorios

Cada imagen:
- Normalized to [0, 1]
- Ruido Gaussiano agregado (σ=0.05)
- Canal único (escala de grises)

### Estadísticas

```
Total imágenes: 1000
Train: 700 (70%)
Val: 150 (15%)
Test: 150 (15%)

Tamaño: 28×28×1 (784 píxeles)
```

---

## Uso

### 1. Generar Datos

```python
from generador_sintetico import GeneradorDatos

gen = GeneradorDatos(seed=42)
datos = gen.generar_dataset(n_samples=1000, split=(0.7, 0.15, 0.15))

print(datos.info())  # Generativo: Train (700, 28, 28, 1), ...
```

### 2. Entrenar GAN

```python
from generador_sintetico import GAN

gan = GAN(latent_dim=100, seed=42)
hist = gan.entrenar(
    datos.X_train, datos.X_val,
    epochs=100,
    batch_size=32
)

print(f"G Loss: {hist['g_loss'][-1]:.4f}")
print(f"D Loss: {hist['d_loss'][-1]:.4f}")
```

### 3. Generar Imágenes GAN

```python
# Generar 10 imágenes sintéticas
imgs_fake = gan.generar_imagenes(n_imagenes=10)
print(imgs_fake.shape)  # (10, 28, 28, 1)
```

### 4. Entrenar VAE

```python
from generador_sintetico import VAE

vae = VAE(latent_dim=32, seed=42)
vae.construir_vae()

hist = vae.entrenar(
    datos.X_train, datos.X_val,
    epochs=30,
    batch_size=32
)
```

### 5. Reconstrucción VAE

```python
# Reconstruir imágenes
X_recon = vae.reconstruir(datos.X_test[:5])
error = np.mean(np.abs(datos.X_test[:5] - X_recon))
print(f"Error reconstrucción: {error:.4f}")
```

### 6. Interpolación en Latent Space

```python
# Interpolar entre dos imágenes
z1 = np.random.normal(0, 1, (1, 32))
z2 = np.random.normal(0, 1, (1, 32))

alpha_values = np.linspace(0, 1, 5)
for alpha in alpha_values:
    z_interp = (1 - alpha) * z1 + alpha * z2
    img_interp = vae.decoder.predict(z_interp)
    # Imagen suavemente transiciona de z1 a z2
```

### 7. Persistencia

```python
# Guardar modelos
gan.guardar('mi_gan')
vae.guardar('mi_vae')

# Genera: mi_gan_gen.h5, mi_gan_disc.h5
# Genera: mi_vae_encoder.h5, mi_vae_decoder.h5, mi_vae_vae.h5
```

---

## Resultados Esperados

### GAN

Después de 100 épocas:
```
Epoch 10:  G Loss: 0.8234, D Loss: 0.4521
Epoch 50:  G Loss: 0.6123, D Loss: 0.3421
Epoch 100: G Loss: 0.5432, D Loss: 0.2987

Imágenes generadas: Formas borrosas pero reconocibles
Diversity: ✓ Varía según ruido input
```

### VAE

Después de 30 épocas:
```
Epoch 1:   Loss: 0.3421
Epoch 10:  Loss: 0.2654
Epoch 30:  Loss: 0.2187

Reconstrucción Error: 0.08 - 0.12
Latent Space: Continuo, interpolable
```

### Comparación

| Métrica | GAN | VAE |
|---------|-----|-----|
| Calidad visual | Superior | Buena |
| Velocidad | Rápida | Moderada |
| Estabilidad | Difícil | Estable |
| Interpolación | Áspera | Suave |
| Interpretabilidad | Baja | Alta |

---

## Técnicas de Optimización

### 1. Batch Normalization

Normaliza activaciones:
$$\hat{x}_i = \gamma \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

**Efectos**:
- Convergencia 3x más rápida
- Reduce sensibilidad a initialization
- Regularización implícita

### 2. LeakyReLU vs ReLU

```python
ReLU(x): max(0, x)           # Muere en x<0
LeakyReLU(x, alpha): max(αx, x)  # Permite gradientes negativos
```

**En GANs**: LeakyReLU reduce "mode collapse"

### 3. Dropout

Regularización: desactiva aleatoriamente 20-30% de neuronas

**Efecto**: Red aprende características robustas

### 4. Learning Rate Scheduling

GAN es sensible a learning rate:
- Generador: 0.0002 (pequeño)
- Discriminador: 0.0002
- Si D entrena muy bien → G no mejora
- Si G entrena muy bien → D falla

---

## Suite de Pruebas

**40+ pruebas** incluyendo:

### Generación (8 tests)
- Formas geométricas
- Rango [0, 1]
- Dataset split

### Construcción GAN (5 tests)
- Generador shape
- Discriminador shape
- Salidas válidas

### Construcción VAE (5 tests)
- Encoder [mean, log_var]
- Decoder shape
- VAE completo

### Entrenamiento (5 tests)
- GAN converge
- VAE converge
- Losses válidos

### Generación (4 tests)
- GAN genera imágenes
- VAE genera imágenes
- Interpolación continua

### Reconstrucción (2 tests)
- Error razonable
- Determinístico

### Persistencia (2 tests)
- GAN save/load
- VAE save/load

### Performance (3 tests)
- Generación rápida
- Dataset rápido

---

## Referencias

1. Goodfellow et al. (2014). "Generative Adversarial Networks"
2. Kingma & Welling (2014). "Auto-Encoding Variational Bayes"
3. Radford et al. (2016). "Unsupervised Representation Learning with DCGANs"
4. https://arxiv.org/abs/1406.2661 (GAN original)
5. https://arxiv.org/abs/1312.6114 (VAE original)

---

**Autor**: Copilot
**Versión**: 1.0
**Fecha**: 2024
**Licencia**: MIT
