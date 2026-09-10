# Clasificación de Dígitos con KNN sobre el dataset MNIST

Proyecto de la asignatura **Métodos y Modelos Matemáticos para IA**. Implementa un clasificador de dígitos escritos a mano (0-9) usando el algoritmo **K-Nearest Neighbors (KNN)**, entrenado sobre el dataset **MNIST**, con una interfaz web que permite clasificar dígitos escritos a mano en tiempo real usando la cámara del computador o del celular.

**Integrantes:** Juan Andrés Correa Arenas, Juan José López Valencia, Isabella Garzón Salazar

---

## 📋 Descripción del proyecto

El modelo no usa los píxeles crudos de las imágenes directamente. En su lugar:

1. **Normaliza geométricamente** cada dígito: lo recorta al área donde hay trazo, lo redimensiona preservando su proporción y lo centra según su centro de masa en un lienzo de 28×28 (imitando el proceso con el que se construyó MNIST originalmente).
2. **Extrae características de gradiente** (similares a HOG — Histogram of Oriented Gradients): calcula gradientes con el operador de Sobel y arma histogramas de orientación en una cuadrícula 4×4, resultando en **144 características por imagen** en vez de las 784 originales.
3. Clasifica usando **KNN con distancia euclidiana y voto ponderado por distancia** (los vecinos más cercanos pesan más en la votación).

Esto hace que el modelo sea más robusto frente a variaciones de grosor de trazo e iluminación al clasificar fotografías reales, no solo imágenes ya limpias de MNIST.

---

## 🗂️ Estructura del repositorio

├── app.py # Interfaz web (Streamlit) con captura por cámara
├── main.py # Versión de línea de comandos (CLI)
├── mnist_model.py # Lógica central: normalización, extracción de features, KNN
├── test_model.py # Script de evaluación (accuracy, matriz de confusión, baseline)
├── img/ # Imágenes propias de prueba (6 fotos por dígito, en subcarpetas 0-9)
├── requirements.txt # Dependencias del proyecto
├── Dockerfile # Imagen para correr la app en un contenedor
└── README.md


---

## ⚙️ Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/juancorrea22/MNIST-Met.-Mod.-Mat.-para-IA-2026-2.git
cd MNIST-Met.-Mod.-Mat.-para-IA-2026-2

# 2. Crear y activar un entorno virtual
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\activate
# Windows (Git Bash):
source venv/Scripts/activate
# Mac/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Uso

### Interfaz web (recomendado — permite usar la cámara)

```bash
streamlit run app.py
```

Se abre en el navegador en `http://localhost:8501`. Permite tomar una foto de un dígito escrito a mano o subir una imagen, ajustar el valor de K y el número de muestras de entrenamiento en tiempo real, y ver la predicción junto con el nivel de confianza y la distancia a los vecinos más cercanos.

### Línea de comandos (CLI)

```bash
python main.py
```

Clasifica imágenes guardadas localmente, indicando la ruta del archivo y el valor de K por consola.

### Evaluación del modelo

```bash
python test_model.py
```

Entrena con 500 imágenes por dígito y evalúa contra 100 imágenes distintas por dígito. Genera:
- Accuracy global y por dígito.
- Matriz de confusión (impresa en consola y guardada como `matriz_confusion.png`).
- Tiempos de predicción.
- Comparación contra un modelo baseline.
- Reporte de clasificación completo (precisión, recall, F1).

### Con Docker

```bash
docker build -t mnist-knn .
docker run -p 8501:8501 mnist-knn
```

---

## 📊 Resultados

Evaluado sobre 1000 imágenes de prueba de MNIST (100 por dígito, no usadas en entrenamiento):

| Métrica | Valor |
|---|---|
| Accuracy global | 91.6% |
| Tiempo promedio por predicción | 0.11 ms |
| Accuracy del baseline (clase más frecuente) | 10.0% |

La mayor confusión del modelo ocurre entre los dígitos **7 y 9** (13% de los casos), por la similitud de sus trazos curvos al representarse como características de gradiente. Ver `matriz_confusion.png` para el detalle completo.

---
