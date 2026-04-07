import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import os
import zipfile
import shutil

# --- Step 1: Preprocess your data from the downloaded Kaggle zip file ---
def load_and_preprocess_data(zip_path="archive.zip", image_size=(80, 80)):
    """
    Loads and preprocesses "no-ship" images from the Kaggle dataset.
    """
    images = []
    temp_dir = "temp_data_extraction"
    
    if not os.path.exists(zip_path):
        print(f"Error: The file '{zip_path}' was not found.")
        print("Please download the 'Ships in Satellite Imagery' dataset from Kaggle and rename it to 'archive.zip'.")
        return None

    try:
        print(f"Extracting data from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # CORRECTED: The path to the images is nested within two 'shipsnet' folders.
        image_dir = os.path.join(temp_dir, "shipsnet", "shipsnet")
        if not os.path.exists(image_dir):
            print(f"Error: The expected directory '{image_dir}' was not found inside the zip file.")
            print("Please ensure your downloaded zip file has a nested 'shipsnet/shipsnet' folder.")
            return None

        print("Loading 'no-ship' images for training...")
        for filename in os.listdir(image_dir):
            # The '0_' prefix indicates a "ship" image, so we load all others as "no-ship"
            if not filename.startswith('0_'): 
                img_path = os.path.join(image_dir, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                img_array = np.array(img).astype('float32') / 255.0
                images.append(img_array)
        
        if not images:
            print("No 'no-ship' images found. Please check the dataset contents.")
            return None

        data = np.array(images)
        print(f"Successfully loaded {data.shape[0]} 'no-ship' images of size {data.shape[1]}x{data.shape[2]}x{data.shape[3]}.")
        return data

    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return None
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory '{temp_dir}'.")

# --- Step 2: Define the Convolutional Autoencoder Model ---
def build_cae(input_shape=(80, 80, 3)):
    """
    Builds a Convolutional Autoencoder model for 80x80 RGB images.
    """
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Latent space representation to rebuild the images 
    latent_space = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(latent_space)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(encoder_input, decoder_output)
    return model

# --- Step 3: Train the Model ---
def train_model(data, model_path='cae_model.h5'):
    """
    Trains the autoencoder model on the provided data.
    """
    model = build_cae(input_shape=data.shape[1:])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True)
    
    print("\nStarting model training...")
    history = model.fit(
        data,
        data,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )
    print(f"Model saved to {model_path}")
    return model

if __name__ == "__main__":
    normal_images = load_and_preprocess_data()
    
    if normal_images is not None:
        trained_model = train_model(normal_images)
        
        print("\nCalculating anomaly threshold...")
        reconstructions = trained_model.predict(normal_images)
        reconstruction_errors = np.mean(np.square(normal_images - reconstructions), axis=(1, 2, 3))
        threshold = np.percentile(reconstruction_errors, 95)
        np.save("anomaly_threshold.npy", np.array([threshold]))
        print(f"Anomaly threshold saved: {threshold}")
    else:
        print("Data loading failed. Exiting.")