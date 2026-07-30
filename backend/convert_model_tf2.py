
#############################################
#  Model Rebuild uisng keras (not important)
############################################







import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# distance function used in siamese
def l1_distance(vects):
    x, y = vects
    return K.abs(x - y)

# load model WITHOUT lambda
model = tf.keras.models.load_model(
    "model/signature_siamese_model1.h5",
    compile=False,
    custom_objects={"l1_distance": l1_distance}
)

model.save("model/signature_siamese_model1.keras")

print("Model rebuilt and saved")