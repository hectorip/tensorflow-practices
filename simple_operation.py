import tensorflow as tf

# Definir un grafo de cómputo simple
@tf.function
def simple_computation(x, y):
    return tf.add(x, y, name="add")

# Crear un escritor de resumen para TensorBoard
log_dir = "logs/simple_graph"

# Inicia el perfilador
tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)

# Registrar el grafo de cómputo
with tf.summary.create_file_writer(log_dir).as_default():
    # Ejecutar la función para registrar el grafo
    result = simple_computation(tf.constant(4.0), tf.constant(2.0))

    tf.summary.trace_export(name="simple_graph", step=10)