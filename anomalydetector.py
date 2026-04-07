import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# --- Configuration ---
MODEL_PATH = 'cae_model.h5'
# The training script saved it as anomaly_threshold.npy, let's check both
THRESHOLD_TXT = 'threshold.txt'
THRESHOLD_NPY = 'anomaly_threshold.npy'
IMG_SIZE = (80, 80)

def get_anomaly_threshold():
    """
    Reads the threshold calculated during your latest training.
    """
    # 1. Try reading from the .npy file (which your training script just saved)
    if os.path.exists(THRESHOLD_NPY):
        try:
            val = np.load(THRESHOLD_NPY)
            return float(val[0])
        except:
            pass
            
    # 2. Try reading from the .txt file (the other common format)
    if os.path.exists(THRESHOLD_TXT):
        try:
            with open(THRESHOLD_TXT, 'r') as f:
                return float(f.read().strip())
        except:
            pass

    # 3. Fallback only if files are missing
    return 0.00117 

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        # compile=False makes loading faster and avoids error with custom metrics
        return load_model(MODEL_PATH, compile=False)
    return None

def detect_anomaly(uploaded_file, model, threshold):
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized).astype('float32') / 255.0
        input_batch = np.expand_dims(img_array, axis=0)

        reconstructed_batch = model.predict(input_batch)
        reconstructed_img = reconstructed_batch[0]

        # MSE Calculation
        error_map = np.mean(np.square(img_array - reconstructed_img), axis=-1)
        anomaly_mask = np.where(error_map > threshold, 1.0, 0.0)

        return img_array, reconstructed_img, error_map, anomaly_mask
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None