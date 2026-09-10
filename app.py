"""
Interfaz web para clasificación de dígitos MNIST.

Uso en la demo en vivo:
1. El usuario escribe un dígito (0-9) en un papel, con trazo grueso y oscuro
   sobre fondo claro.
2. Toma una foto con la cámara del PC o del celular directamente en la app
   (botón "Take Photo" del widget de cámara).
3. La app preprocesa la imagen (escala de grises, 28x28, inversión de colores,
   normalización) y el modelo KNN entrenado con MNIST predice el dígito.
4. Se muestra la predicción, el nivel de confianza (votos de los k vecinos)
   y una vista de cómo quedó la imagen luego del preprocesamiento.

Ejecutar localmente:
    streamlit run app.py
"""

import numpy as np
import streamlit as st
from PIL import Image

from mnist_model import load_mnist_subset, train_knn, preprocess_image, predict_digit

st.set_page_config(page_title="Clasificador de Dígitos MNIST", page_icon="🔢", layout="centered")

st.title("Clasificador de Dígitos MNIST (KNN)")
st.write(
    "Toma una foto de un dígito escrito a mano (0-9) usando la cámara y el "
    "modelo KNN, entrenado sobre el dataset MNIST, lo va a clasificar."
)

# --- Barra lateral: parámetros del modelo ---
st.sidebar.header("Parámetros del modelo")
samples_per_class = st.sidebar.slider(
    "Muestras de entrenamiento por dígito", min_value=10, max_value=200, value=40, step=10
)
k = st.sidebar.slider("Valor de K (vecinos)", min_value=1, max_value=15, value=5, step=1)

st.sidebar.caption(
    "Aumentar las muestras por dígito mejora la precisión pero hace más lento "
    "el entrenamiento inicial y la predicción."
)


@st.cache_resource(show_spinner="Descargando MNIST y entrenando el modelo...")
def get_trained_model(samples_per_class: int, k: int):
    train_features, train_labels = load_mnist_subset(samples_per_class=samples_per_class)
    model = train_knn(train_features, train_labels, k=k)
    return model


model = get_trained_model(samples_per_class, k)

st.divider()
st.subheader("Captura del dígito")

camera_photo = st.camera_input(
    "Apunta la cámara al dígito escrito a mano y toma la foto"
)

uploaded_file = st.file_uploader(
    "…o si prefieres, sube una imagen ya tomada", type=["png", "jpg", "jpeg"]
)

image_source = camera_photo or uploaded_file

if image_source is not None:
    pil_image = Image.open(image_source)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Imagen original", width=200)

    image_array = preprocess_image(pil_image)

    with col2:
        # Se reconstruye la imagen 28x28 preprocesada para mostrar qué "ve" el modelo
        preview = (image_array.reshape(28, 28) * 255).astype(np.uint8)
        st.image(preview, caption="Imagen procesada (28x28, invertida)", width=200)

    prediction, distances, neighbor_idx, confidence, votes = predict_digit(model, image_array, k)
    neighbor_labels = model._y[neighbor_idx]

    st.divider()
    st.subheader("Resultado")
    st.metric(label="Dígito predicho", value=str(prediction))
    st.progress(
    float(confidence),
    text=f"Confianza (votos de los {k} vecinos): {float(confidence) * 100:.0f}%"
)
    with st.expander("Detalles técnicos de la predicción"):
        st.write(f"**Vecinos más cercanos:** {neighbor_labels}")
        st.write(f"**Distancias euclidianas:**")
        st.write(np.round(distances, 2))

        st.write("**Votos por clase (0-9):**")
        st.bar_chart(votes)
else:
    st.info("Esperando una foto o imagen para clasificar…")

st.divider()
st.caption(
    "Métodos y Modelos Matemáticos para IA · Mini-Proyecto: Clasificación de dígitos MNIST con KNN"
)