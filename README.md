# Importar bibliotecas
import tensorflow as tf
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o conjunto de dados Iris
iris = load_iris()

X = iris.data
y = iris.target

print("Dados carregados com sucesso!")
print("Quantidade de amostras:", len(X))

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Normalizar os dados
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Dados preparados!")

# Construir o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Configurar o modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Mostrar o modelo
model.summary()

# Treinar o modelo
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=10,
    validation_split=0.1
)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)

print("\nPrecisão do modelo:", accuracy)
print("Precisão em porcentagem:", accuracy * 100, "%")

# Fazer previsões
predictions = model.predict(X_test)

# Identificar as classes previstas
y_pred = predictions.argmax(axis=1)

print("\nPrevisões:")
print(y_pred)

print("\nValores reais:")
print(y_test)

# Testar uma nova flor
nova_flor = [[5.1, 3.5, 1.4, 0.2]]

# Normalizar a nova flor
nova_flor = scaler.transform(nova_flor)

# Fazer previsão
previsao = model.predict(nova_flor)

# Identificar a classe
classe = previsao.argmax(axis=1)[0]

print("\nNova flor:")
print("Classe prevista:", classe)
print("Espécie prevista:", iris.target_names[classe])
