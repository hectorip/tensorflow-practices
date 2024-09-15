import tensorflow as tf
import datetime
import numpy as np

# Crear datos dummy
x_train = np.random.rand(100, 4)  # 100 muestras, cada una con 4 características
y_train = np.random.rand(100, 1)  # 100 valores de salida

# Definir el modelo
class PerceptronModel(tf.Module):
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([4, 1], dtype=tf.float64), name='weight')  # Cambiado a float64
        self.b = tf.Variable(tf.zeros([1], dtype=tf.float64), name='bias')  # Cambiado a float64

    def __call__(self, x):
        x = tf.cast(x, tf.float64)  # Asegurarse de que x sea float64
        return tf.matmul(x, self.w) + self.b

# Crear el modelo
model = PerceptronModel()

# Definir la pérdida y el optimizador
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.optimizers.Adam()

# Función de entrenamiento
def train_step(model, inputs, labels, loss_object, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Configurar TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)

# Entrenar el modelo
epochs = 10
for epoch in range(epochs):
    loss = train_step(model, x_train, y_train, loss_object, optimizer)
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=epoch)
    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')