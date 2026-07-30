import tensorflow as tf
import numpy as np
import os

# ---------------------------------------------------
# Model Path
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "signature_siamese_model1.keras")

print("Loading Siamese Model...")

# ---------------------------------------------------
# Custom Layers
# ---------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


@tf.keras.utils.register_keras_serializable()
class EuclideanDistance(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

siamese_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "L2Normalization": L2Normalization,
        "EuclideanDistance": EuclideanDistance
    },
    compile=False
)

print("Siamese Model Loaded Successfully")


# ---------------------------------------------------
# Extract Embedding Network
# ---------------------------------------------------

embedding_model = siamese_model.get_layer("functional")

print("Embedding model extracted")


# ---------------------------------------------------
# Get Embedding
# ---------------------------------------------------

def get_embedding(image):

    image_batch = np.expand_dims(image, axis=0)

    embedding = embedding_model.predict(image_batch, verbose=0)

    return embedding[0]


# ---------------------------------------------------
# Euclidean Distance
# ---------------------------------------------------

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)