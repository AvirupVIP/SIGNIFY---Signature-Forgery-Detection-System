import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_signature(image_input, from_bytes=False):

    # Load image
    if from_bytes:
        file_bytes = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image could not be loaded")

    # Binarization
    _, img_bin = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

    # Crop signature
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        raise ValueError("No signature detected")

    x, y, w, h = cv2.boundingRect(coords)
    img_crop = img_bin[y:y+h, x:x+w]

    # Resize
    img_resized = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))

    # Normalize
    normalized = img_resized.astype("float32") / 255.0

    # Convert 1 channel → 3 channels
    normalized = np.stack((normalized,)*3, axis=-1)

    return normalized, img_bin