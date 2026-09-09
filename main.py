"""
main.py
-------
Versión de línea de comandos (CLI) del clasificador de dígitos MNIST.
Útil para pruebas rápidas por consola. Para la demo en vivo con cámara,
usar 'streamlit run app.py' (ver README.md).
"""

import os
from PIL import Image

from mnist_model import load_mnist_subset, train_knn, preprocess_image, predict_digit

print("--- LOADING AND PREPARING MNIST DATASET ---")

samples_per_class = 40  # Número de imágenes que se van a usar por cada dígito
train_features, train_labels = load_mnist_subset(samples_per_class=samples_per_class)

print(f"Model data ready. Total training samples: {train_features.shape[0]}")
print("==================================================")

is_running = True
while is_running:
    print("\n--- NEW CLASSIFICATION TEST ---")
    image_name = input("Enter the test image name (e.g., my_digit.png) or type 'exit': ").strip()

    if image_name.lower() == 'exit':
        print("Exiting program!")
        is_running = False
        continue

    if not os.path.exists(image_name):
        print(f"[Error] The file '{image_name}' was not found in the current directory.")
        continue

    try:
        image = Image.open(image_name)
        image_array = preprocess_image(image)

        k = int(input("Enter the value of K (e.g., 3, 5, 7): ").strip())
        model = train_knn(train_features, train_labels, k=k)

        prediction, distances, neighbor_idx, confidence, votes = predict_digit(model, image_array, k)

        print(f"\n[RESULT] The model (with K={k}) predicts the digit is: **{prediction}**")
        print(f"Confidence (neighbor vote share): {confidence * 100:.0f}%")
        print(f"Distances to the {k} nearest neighbors: {distances.round(2)}")

    except Exception as error:
        print(f"Error processing the image: {error}")