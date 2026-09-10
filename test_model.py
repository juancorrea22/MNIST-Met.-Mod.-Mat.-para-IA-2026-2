"""

Evalúa el modelo KNN usando exactamente el mismo pipeline que la app real
(normalize_digit + extract_features), con un split real de entrenamiento
sobre MNIST.

Genera:
- Accuracy global y por dígito.
- Matriz de confusión (tabla + gráfico guardado como PNG).
- Tiempos de predicción.
- Comparación contra un modelo base (baseline).
- Reporte de clasificación completo.

"""

import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from mnist_model import train_knn, normalize_digit, extract_features

print("Descargando MNIST...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

# --- 1. Separar entrenamiento y prueba (imágenes distintas, sin traslape) ---
SAMPLES_TRAIN = 500
SAMPLES_TEST = 100

train_indices_all = []
test_indices_all = []

for digit in range(10):
    indices = np.where(y == digit)[0]
    train_indices_all.append(indices[:SAMPLES_TRAIN])
    test_indices_all.append(indices[SAMPLES_TRAIN:SAMPLES_TRAIN + SAMPLES_TEST])

train_indices_all = np.concatenate(train_indices_all)
test_indices_all = np.concatenate(test_indices_all)

train_y = y[train_indices_all]
test_y = y[test_indices_all]

print(f"Entrenamiento: {len(train_y)} imágenes")
print(f"Prueba: {len(test_y)} imágenes")

# --- 2. Aplicar EL MISMO pipeline que usa la app: normalizar + extraer features ---
# Esto es clave: si aquí se usaran los píxeles crudos, se estaría evaluando un
# modelo distinto al que realmente corre en app.py.


def build_features(indices):
    features_list = []
    for idx in indices:
        image_28x28 = X[idx].reshape(28, 28)
        normalized = normalize_digit(image_28x28)
        features = extract_features(normalized)
        features_list.append(features)
    return np.array(features_list, dtype=np.float32)


print("\nExtrayendo características de entrenamiento (puede tardar unos minutos)...")
train_X = build_features(train_indices_all)

print("Extrayendo características de prueba...")
test_X = build_features(test_indices_all)

# --- 3. Entrenar KNN (mismo modelo que usa la app) ---
k = 5
print(f"\nEntrenando KNN con K={k}...")
model = train_knn(train_X, train_y, k=k)

# --- 4. Predicciones y tiempos ---
print("Probando modelo...")

inicio = time.perf_counter()
predictions = model.predict(test_X)
fin = time.perf_counter()

tiempo_total = fin - inicio
tiempo_promedio_ms = (tiempo_total / len(test_y)) * 1000

# --- 5. Accuracy general ---
accuracy = accuracy_score(test_y, predictions)

print("\n==============================")
print("RESULTADOS")
print("==============================")
print(f"\nAccuracy total: {accuracy * 100:.2f}%")
print(f"Tiempo total de predicción ({len(test_y)} imágenes): {tiempo_total:.4f} s")
print(f"Tiempo promedio por imagen: {tiempo_promedio_ms:.2f} ms")

# --- 6. Accuracy por número ---
print("\nAccuracy por dígito:\n")
for digit in range(10):
    mask = test_y == digit
    digit_accuracy = accuracy_score(test_y[mask], predictions[mask])
    print(f"{digit}: {digit_accuracy * 100:.2f}%")

# --- 7. Matriz de confusión ---
cm = confusion_matrix(test_y, predictions, labels=list(range(10)))

print("\n==============================")
print("MATRIZ DE CONFUSIÓN")
print("==============================")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicción del modelo")
plt.ylabel("Clase real")
plt.title(f"Matriz de Confusión — KNN con features de gradiente (k={k})")
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=150)
print("\nGráfico guardado como 'matriz_confusion.png' (para pegar en la diapositiva).")

# --- 8. Comparación contra baseline ---
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(train_X, train_y)
baseline_predictions = baseline.predict(test_X)
baseline_accuracy = accuracy_score(test_y, baseline_predictions)

print("\n==============================")
print("COMPARACIÓN CONTRA BASELINE")
print("==============================")
print(f"Accuracy del baseline (clase más frecuente): {baseline_accuracy * 100:.2f}%")
print(f"Accuracy del modelo KNN (k={k}): {accuracy * 100:.2f}%")
print(f"Mejora sobre el baseline: +{(accuracy - baseline_accuracy) * 100:.2f} puntos porcentuales")

# --- 9. Reporte completo ---
print("\n==============================")
print("REPORTE COMPLETO")
print("==============================\n")
print(classification_report(test_y, predictions, digits=4))