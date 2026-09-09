import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report

from mnist_model import train_knn


print("Descargando MNIST...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="auto"
)

X = mnist.data / 255.0
y = mnist.target.astype(int)


# --------------------------------------------------
# Separar entrenamiento y prueba
# --------------------------------------------------

# Tomamos 500 imágenes por dígito para entrenamiento
# y otras imágenes diferentes para prueba.

train_X = []
train_y = []

test_X = []
test_y = []


for digit in range(10):

    indices = np.where(y == digit)[0]

    # primeras 500 para entrenar
    train_indices = indices[:500]

    # siguientes 100 para probar
    test_indices = indices[500:600]

    train_X.append(X[train_indices])
    train_y.append(y[train_indices])

    test_X.append(X[test_indices])
    test_y.append(y[test_indices])


train_X = np.vstack(train_X)
train_y = np.concatenate(train_y)

test_X = np.vstack(test_X)
test_y = np.concatenate(test_y)


print(f"Entrenamiento: {len(train_y)} imágenes")
print(f"Prueba: {len(test_y)} imágenes")


# --------------------------------------------------
# Entrenar KNN
# --------------------------------------------------

k = 5

print(f"\nEntrenando KNN con K={k}...")

model = train_knn(
    train_X,
    train_y,
    k=k
)


# --------------------------------------------------
# Hacer predicciones
# --------------------------------------------------

print("Probando modelo...")

predictions = model.predict(test_X)


# --------------------------------------------------
# Accuracy general
# --------------------------------------------------

accuracy = accuracy_score(
    test_y,
    predictions
)

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(
    f"\nAccuracy total: {accuracy * 100:.2f}%"
)


# --------------------------------------------------
# Accuracy por número
# --------------------------------------------------

print("\nAccuracy por dígito:\n")

for digit in range(10):

    mask = test_y == digit

    digit_accuracy = accuracy_score(
        test_y[mask],
        predictions[mask]
    )

    print(
        f"{digit}: {digit_accuracy * 100:.2f}%"
    )


# --------------------------------------------------
# Reporte completo
# --------------------------------------------------

print("\n==============================")
print("REPORTE COMPLETO")
print("==============================\n")

print(
    classification_report(
        test_y,
        predictions,
        digits=4
    )
)