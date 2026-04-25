import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#insira seu código aqui
import os
import numpy as np

print("Carregando o dataset MNIST:")

# Vai tenta ler local  e caso não ache vai baixar
if os.path.exists('mnist.npz'):
    print("Arquivo mnist.npz encontrado localmente. Lendo offline:")
    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
else:
    print("Arquivo não encontrado localmente. Lendo Online:")
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizanodo os dados para o intervalo [0, 1]
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print("Construindo o modelo CNN:")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Treinando o modelo (limite de 5 épocas para CI:)")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nAcurácia final no conjunto de teste: {accuracy * 100:.2f}%")

print("Salvando o modelo no formato Keras (model.h5):")
model.save('model.h5')
print("Treinamento concluído com sucesso!")