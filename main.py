import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

print("--- LOADING AND PREPARING MNIST DATASET ---")

# descarga de las 70000 imágenes 28 x 28 pixeles del dataset MNIST.
mnist_data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
features, labels = mnist_data.data, mnist_data.target.astype(int)

samples_per_class = 40 # Número de imagenes que se van a usar por cada digito
subset_features = list()
subset_labels = list()

for digit in range(10):
    # np.where devuelve los índices de todas las muestras cuya etiqueta es 'digit'
    indices = np.where(labels == digit)[0]

    # Se toman solo las primeras 'samples_per_class' de esas muestras
    selected_indices = indices[:samples_per_class]
    subset_features.append(features[selected_indices])
    subset_labels.append(labels[selected_indices])

# Se apilan los 10 subconjuntos (uno por dígito) en una sola matriz de entrenamiento
train_features = np.vstack(subset_features)
train_labels = np.concatenate(subset_labels)

# se pasan los valores de los pixeles de números entre 0 y 255 a 0 y 1
train_features = train_features / 255.0

print(f"Model data ready. Total training samples: {train_features.shape[0]}")
print("==================================================")

is_running = True
while is_running:
    print("\n--- NEW CLASSIFICATION TEST ---")
    image_name = input("Enter the test image name (e.g., my_digit.png) or type 'exit': ").strip()
    
    if image_name.lower() == 'exit':
        print("Exiting program!")
        is_running = False

    else:
        if not os.path.exists(image_name):
            print(f"[Error] The file '{image_name}' was not found in the current directory.")
        else:
            try:
                # se pone la imagen que se habra en 28 x 28 pixeles
                image = Image.open(image_name).convert("L").resize((28, 28))
                
                # Inversión de colores para compatibilidad con MNIST
                # Convierte fondo blanco/trazo negro a fondo negro/trazo blanco de las imagenes de nuesta carpeta img
                image = image.point(lambda p: 255 - p)

                # Se aplana la imagen 28 x 28 a un vector de 784 valores
                image_array = np.array(image).reshape(1, -1) / 255.0
                
                k = int(input("Enter the value of K (e.g., 3, 5, 7): ").strip())

                # se realiza un KNN con el k seleccionado y usando distancia euclidiana
                knn_classifier = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
                knn_classifier.fit(train_features, train_labels)

                # genera la predicción por medio de la distancia euclidiana
                prediction = knn_classifier.predict(image_array)

                # se generan las distancias de los k vecinos mas cercanos.
                distances, neighbor_indices = knn_classifier.kneighbors(image_array, n_neighbors=k)
                
                print(f"\n[RESULT] The model (with K={k}) predicts the digit is: **{prediction[0]}**")
                print(f"Distances to the {k} nearest neighbors: {np.round(distances[0], 2)}")
                
            except Exception as error:
                print(f"Error processing the image: {error}")