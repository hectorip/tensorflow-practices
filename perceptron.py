import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import TensorBoard
import datetime
import numpy as np
# Crear datos dummy
x_train = np.random.rand(100, 4)  # 100 muestras, cada una con 4 características
y_train = np.random.rand(100, 1)  # 100 valores de salida

# Crear el modelo
model = Sequential()

# Añade una capa de entrada explícita
#model.add(Input(shape=(4,)))

# Añade una capa densa con una sola neurona
model.add(Dense(1, input_shape=(4,), activation='linear'))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Resumen del modelo
#model.summary()

# Configurar TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Entrenar el modelo con el callback de TensorBoard
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])
