"""
mnist_model.py
----------------
Módulo con la lógica central del proyecto: carga del dataset MNIST,
construcción del subconjunto de entrenamiento, entrenamiento del clasificador
KNN y preprocesamiento de imágenes externas para predicción.

Se separa de main.py / app.py para que tanto la CLI, la interfaz web (Streamlit)
y los scripts de evaluación de métricas reutilicen exactamente la misma lógica.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image


def load_mnist_subset(samples_per_class: int = 40, seed: int = 42):
    """
    Descarga MNIST (784 features = 28x28 px) y arma un subconjunto balanceado
    con 'samples_per_class' imágenes por cada dígito (0-9).

    Retorna (features, labels) ya normalizados a rango [0, 1].
    """
    mnist_data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    features, labels = mnist_data.data, mnist_data.target.astype(int)

    rng = np.random.default_rng(seed)
    subset_features, subset_labels = [], []

    for digit in range(10):
        indices = np.where(labels == digit)[0]
        # Selección aleatoria (con semilla fija) en vez de tomar siempre las primeras N,
        # para que el subconjunto no dependa del orden en que vienen los datos.
        chosen = rng.choice(indices, size=min(samples_per_class, len(indices)), replace=False)
        subset_features.append(features[chosen])
        subset_labels.append(labels[chosen])

    train_features = np.vstack(subset_features) / 255.0
    train_labels = np.concatenate(subset_labels)
    return train_features, train_labels


def train_knn(train_features, train_labels, k: int = 5):
    """Entrena y retorna un KNeighborsClassifier con distancia euclidiana."""
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(train_features, train_labels)
    return model


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convierte una imagen PIL arbitraria al formato esperado por el modelo:
    escala de grises, 28x28 px, colores invertidos (fondo negro / trazo blanco,
    igual que MNIST) y aplanada a un vector de 784 valores normalizados [0, 1].
    """
    image = pil_image.convert("L").resize((28, 28))
    image = image.point(lambda p: 255 - p)
    array = np.array(image).reshape(1, -1) / 255.0
    return array


def predict_digit(model: KNeighborsClassifier, image_array: np.ndarray, k: int):
    """Retorna (predicción, distancias, índices_vecinos, probabilidades_por_clase)."""
    prediction = model.predict(image_array)[0]
    distances, neighbor_indices = model.kneighbors(image_array, n_neighbors=k)

    # "Confianza" simple: proporción de los k vecinos que votaron por la clase ganadora.
    neighbor_labels = model._y[neighbor_indices[0]]
    votes = np.bincount(neighbor_labels, minlength=10)
    confidence = votes[prediction] / k

    return prediction, distances[0], neighbor_indices[0], confidence, votes