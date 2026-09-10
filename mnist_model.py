"""
Clasificador KNN de dígitos escritos a mano.

- carga MNIST
- normaliza las imágenes
- extrae características de forma/bordes
- entrena KNN con distancia euclidiana
- preprocesa fotografías reales
- predice el dígito y muestra vecinos/confianza
"""

import numpy as np

from PIL import Image
from scipy import ndimage
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

# NORMALIZACIÓN DEL DÍGITO
def normalize_digit(digit: np.ndarray) -> np.ndarray:
    """
    Recibe un dígito blanco sobre fondo negro
    y lo lleva a un formato 28x28 parecido a MNIST.
    """

    digit = digit.astype(np.float32)

    if digit.max() > 0:
        digit = digit / digit.max()

    #detectar dónde realmente hay contenido
    coords = np.argwhere(digit > 0.08)

    if coords.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    #límites del dígito
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Recortar
    digit = digit[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    height, width = digit.shape

    # Redimensionar sin deformar
    if height > width:

        new_height = 20

        new_width = max(
            1,
            round(width * 20 / height)
        )

    else:

        new_width = 20

        new_height = max(
            1,
            round(height * 20 / width)
        )

    digit_image = Image.fromarray(
        (digit * 255).astype(np.uint8)
    )

    digit_image = digit_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    resized = np.array(
        digit_image,
        dtype=np.float32
    ) / 255.0

    # Canvas negro 28x28
    canvas = np.zeros(
        (28, 28),
        dtype=np.float32
    )

    y_position = (28 - new_height) // 2
    x_position = (28 - new_width) // 2

    canvas[
        y_position:y_position + new_height,
        x_position:x_position + new_width
    ] = resized

    # Centrar por centro de masa
    center_y, center_x = ndimage.center_of_mass(
        canvas
    )

    if not np.isnan(center_y):

        shift_y = 13.5 - center_y
        shift_x = 13.5 - center_x

        canvas = ndimage.shift(
            canvas,
            shift=(shift_y, shift_x),
            order=1,
            mode="constant",
            cval=0
        )

    return np.clip(
        canvas,
        0,
        1
    )

# EXTRACCIÓN DE CARACTERÍSTICAS

def extract_features(image_28x28: np.ndarray) -> np.ndarray:
    """
    Extrae características de forma usando gradientes.

    Esto hace que KNN dependa menos de pequeñas diferencias
    de grosor o iluminación y más de la forma del número.
    """

    image = image_28x28.astype(
        np.float32
    )

    # Gradientes horizontal y vertical
    gradient_x = ndimage.sobel(
        image,
        axis=1,
        mode="constant"
    )

    gradient_y = ndimage.sobel(
        image,
        axis=0,
        mode="constant"
    )

    # Magnitud del borde
    magnitude = np.hypot(
        gradient_x,
        gradient_y
    )

    # Dirección del borde
    angle = (
        np.arctan2(
            gradient_y,
            gradient_x
        )
        + np.pi
    ) % np.pi

    # Dividir 28x28 en 4x4 celdas de 7x7
    # Cada celda tendrá 9 orientaciones
    features = []

    number_of_bins = 9

    for cell_y in range(4):

        for cell_x in range(4):

            y_start = cell_y * 7
            y_end = y_start + 7

            x_start = cell_x * 7
            x_end = x_start + 7

            cell_magnitude = magnitude[
                y_start:y_end,
                x_start:x_end
            ].ravel()

            cell_angle = angle[
                y_start:y_end,
                x_start:x_end
            ].ravel()

            histogram = np.zeros(
                number_of_bins,
                dtype=np.float32
            )

            bin_indices = np.floor(
                cell_angle
                / np.pi
                * number_of_bins
            ).astype(int)

            bin_indices = np.clip(
                bin_indices,
                0,
                number_of_bins - 1
            )

            for index, value in zip(
                bin_indices,
                cell_magnitude
            ):

                histogram[index] += value

            # Normalizar histograma
            norm = np.linalg.norm(
                histogram
            )

            if norm > 0:
                histogram = (
                    histogram / norm
                )

            features.extend(
                histogram
            )

    return np.array(
        features,
        dtype=np.float32
    )

# CARGA DEL DATASET MNIST
def load_mnist_subset(
    samples_per_class: int = 500,
    seed: int = 42
):
    """
    Descarga MNIST y crea un subconjunto balanceado.

    Cada imagen:
    1. Se normaliza.
    2. Se convierte en características de forma.
    """

    mnist_data = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto"
    )

    raw_features = (
        mnist_data.data.astype(
            np.float32
        )
        / 255.0
    )

    labels = mnist_data.target.astype(
        int
    )

    rng = np.random.default_rng(
        seed
    )

    train_features = []
    train_labels = []

    for digit in range(10):

        indices = np.where(
            labels == digit
        )[0]

        chosen = rng.choice(
            indices,
            size=min(
                samples_per_class,
                len(indices)
            ),
            replace=False
        )

        for index in chosen:

            image = raw_features[
                index
            ].reshape(
                28,
                28
            )

            image = normalize_digit(
                image
            )

            features = extract_features(
                image
            )

            train_features.append(
                features
            )

            train_labels.append(
                digit
            )

    train_features = np.array(
        train_features,
        dtype=np.float32
    )

    train_labels = np.array(
        train_labels,
        dtype=int
    )

    return (
        train_features,
        train_labels
    )


# ENTRENAMIENTO KNN
def train_knn(
    train_features,
    train_labels,
    k: int = 5
):
    """
    Entrena KNN usando distancia euclidiana.

    Los vecinos más cercanos tienen mayor peso.
    """

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean",
        weights="distance"
    )

    model.fit(
        train_features,
        train_labels
    )

    return model

# PREPROCESAMIENTO DE FOTOGRAFÍAS
def preprocess_image(
    pil_image: Image.Image
) -> np.ndarray:
    """
    Convierte una fotografía real en una imagen 28x28
    limpia y compatible con la visualización de app.py.
    """

    # Escala de grises
    image = pil_image.convert(
        "L"
    )

    original = np.array(
        image,
        dtype=np.float32
    )

    # Estimar iluminación del papel
    background = ndimage.gaussian_filter(
        original,
        sigma=15
    )

    # Lo oscuro respecto al fondo es el número
    digit = background - original

    digit[
        digit < 0
    ] = 0

    if digit.max() > 0:

        digit = (
            digit
            / digit.max()
        )

    # Detectar trazo
    mask = digit > 0.12

    # Unir pequeños huecos
    mask = ndimage.binary_closing(
        mask,
        structure=np.ones(
            (3, 3),
            dtype=bool
        )
    )

    # Componentes conectados
    labeled, count = ndimage.label(
        mask
    )

    if count == 0:

        raise ValueError(
            "No se detectó ningún dígito."
        )

    sizes = ndimage.sum(
        mask,
        labeled,
        range(
            1,
            count + 1
        )
    )

    # Elegir componente principal
    best_label = (
        int(
            np.argmax(sizes)
        )
        + 1
    )

    main_component = (
        labeled == best_label
    )

    # Eliminar ruido externo
    digit = np.where(
        main_component,
        digit,
        0
    )

    # Normalizar al formato 28x28
    normalized = normalize_digit(
        digit
    )

    # app.py usa esto para mostrar la imagen procesada
    return normalized.reshape(
        1,
        784
    )


# CONVERSIÓN DE IMAGEN A FEATURES
def _to_features(
    image_array: np.ndarray
) -> np.ndarray:
    """
    Convierte el 1x784 de app.py
    a las características usadas por KNN.
    """

    image = image_array.reshape(
        28,
        28
    )

    features = extract_features(
        image
    )

    return features.reshape(
        1,
        -1
    )

# PREDICCIÓN
def predict_digit(
    model: KNeighborsClassifier,
    image_array: np.ndarray,
    k: int
):
    """
    Predice el dígito usando KNN.

    Retorna:
    - predicción
    - distancias
    - índices de vecinos
    - confianza
    - votos
    """

    # Convertir imagen a características
    feature_array = _to_features(
        image_array
    )

    # Obtener vecinos
    distances, neighbor_indices = model.kneighbors(
        feature_array,
        n_neighbors=k
    )

    distances = distances[0]

    neighbor_indices = (
        neighbor_indices[0]
    )

    # Etiquetas de los vecinos
    neighbor_labels = model._y[
        neighbor_indices
    ]
    
    # Voto ponderado por distancia
    weights = 1.0 / (
        distances + 1e-8
    )

    votes = np.zeros(
        10,
        dtype=np.float32
    )

    for label, weight in zip(
        neighbor_labels,
        weights
    ):

        votes[
            int(label)
        ] += float(weight)

    # Clase ganadora
    prediction = int(
        np.argmax(votes)
    )

    # Confianza
    total_weight = float(
        votes.sum()
    )

    if total_weight > 0:

        confidence = float(
            votes[prediction]
            / total_weight
        )

    else:

        confidence = 0.0

    return (
        prediction,
        distances,
        neighbor_indices,
        confidence,
        votes
    )