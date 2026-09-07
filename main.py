import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

print("--- LOADING AND PREPARING MNIST DATASET ---")
mnist_data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
features, labels = mnist_data.data, mnist_data.target.astype(int)

samples_per_class = 40
subset_features = list()
subset_labels = list()

for digit in range(10):
    indices = np.where(labels == digit)[0]
    selected_indices = indices[:samples_per_class]
    subset_features.append(features[selected_indices])
    subset_labels.append(labels[selected_indices])

train_features = np.vstack(subset_features)
train_labels = np.concatenate(subset_labels)
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
                image = Image.open(image_name).convert("L").resize((28, 28))
                
                # --- INVERSIÓN DE COLORES PARA COMPATIBILIDAD CON MNIST ---
                # Convierte fondo blanco/trazo negro a fondo negro/trazo blanco
                image = image.point(lambda p: 255 - p)
                
                image_array = np.array(image).reshape(1, -1) / 255.0
                
                k_input = input("Enter the value of K (e.g., 3, 5, 7): ").strip()
                k_value = int(k_input)
                
                knn_classifier = KNeighborsClassifier(n_neighbors=k_value, metric='euclidean')
                knn_classifier.fit(train_features, train_labels)
                
                prediction = knn_classifier.predict(image_array)
                distances, neighbor_indices = knn_classifier.kneighbors(image_array, n_neighbors=k_value)
                
                print(f"\n[RESULT] The model (with K={k_value}) predicts the digit is: **{prediction[0]}**")
                print(f"Distances to the {k_value} nearest neighbors: {np.round(distances[0], 2)}")
                
            except Exception as error:
                print(f"[Error processing the image]: {error}")