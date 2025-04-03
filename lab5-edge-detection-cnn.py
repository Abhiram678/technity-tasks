import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from glob import glob
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_edge_detection_model(input_shape=(256, 256, 3)):
    """
    Create a CNN model for edge detection (based on U-Net architecture)
    
    Parameters:
    input_shape (tuple): Shape of input images
    
    Returns:
    tf.keras.Model: CNN model for edge detection
    """
    inputs = Input(input_shape)
    
    # Encoder (downsampling path)
    # First block
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second block
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Third block
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    
    # Decoder (upsampling path)
    # First block
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = Conv2D(128, (2, 2), activation='relu', padding='same')(up1)
    merge1 = Concatenate()([conv2, up1])
    deconv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
    deconv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(deconv1)
    
    # Second block
    up2 = UpSampling2D(size=(2, 2))(deconv1)
    up2 = Conv2D(64, (2, 2), activation='relu', padding='same')(up2)
    merge2 = Concatenate()([conv1, up2])
    deconv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
    deconv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(deconv2)
    
    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid')(deconv2)
    
    # Create model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

def create_simple_edge_detection_model(input_shape=(256, 256, 3)):
    """
    Create a simpler CNN model for edge detection
    
    Parameters:
    input_shape (tuple): Shape of input images
    
    Returns:
    tf.keras.Model: CNN model for edge detection
    """
    model = Sequential([
        # Input layer
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Hidden layers
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        
        # Output layer
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess image for CNN input
    
    Parameters:
    image_path (str): Path to the input image
    target_size (tuple): Target size for resizing
    
    Returns:
    numpy.ndarray: Preprocessed image
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def generate_edge_map(image, method='canny'):
    """
    Generate edge map for an image using traditional methods
    
    Parameters:
    image (numpy.ndarray): Input image
    method (str): Method to use ('canny', 'sobel', or 'laplacian')
    
    Returns:
    numpy.ndarray: Edge map
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize if needed
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    
    if method == 'canny':
        # Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
    elif method == 'sobel':
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        # Normalize
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = edges.astype(np.uint8)
    elif method == 'laplacian':
        # Laplacian edge detection
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        # Convert back to uint8
        edges = np.uint8(np.absolute(edges))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to [0, 1] for the model training
    edges = edges.astype('float32') / 255.0
    
    return edges

def prepare_training_data(image_dir, edge_method='canny', target_size=(256, 256), sample_limit=None):
    """
    Prepare training data from a directory of images
    
    Parameters:
    image_dir (str): Directory containing input images
    edge_method (str): Method to generate edge maps
    target_size (tuple): Target size for resizing
    sample_limit (int): Maximum number of samples to process
    
    Returns:
    tuple: (input_images, edge_maps)
    """
    # Find all image files
    image_files = glob(os.path.join(image_dir, "*.jpg")) + \
                  glob(os.path.join(image_dir, "*.jpeg")) + \
                  glob(os.path.join(image_dir, "*.png"))
    
    # Limit the number of samples if specified
    if sample_limit:
        image_files = image_files[:sample_limit]
    
    print(f"Found {len(image_files)} images")
    
    # Initialize lists for inputs and targets
    input_images = []
    edge_maps = []
    
    # Process each image
    for image_file in image_files:
        try:
            # Load and resize image
            img = cv2.imread(image_file)
            img = cv2.resize(img, target_size)
            
            # Generate edge map
            edge_map = generate_edge_map(img, method=edge_method)
            
            # Normalize image
            img = img.astype('float32') / 255.0
            
            # Add to lists
            input_images.append(img)
            edge_maps.append(edge_map)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Convert lists to numpy arrays
    input_images = np.array(input_images)
    edge_maps = np.array(edge_maps)
    
    # Add channel dimension to edge maps
    edge_maps = np.expand_dims(edge_maps, axis=-1)
    
    return input_images, edge_maps

def train_edge_detection_model(image_dir, model_path='edge_detection_model.h5', 
                              edge_method='canny', batch_size=8, epochs=10,
                              target_size=(256, 256), sample_limit=None):
    """
    Train a CNN model for edge detection
    
    Parameters:
    image_dir (str): Directory containing training images
    model_path (str): Path to save the trained model
    edge_method (str): Method to generate edge maps for training
    batch_size (int): Batch size for training
    epochs (int): Number of epochs to train
    target_size (tuple): Target size for input images
    sample_limit (int): Maximum number of samples to use
    
    Returns:
    tf.keras.Model: Trained model
    """
    # Prepare training data
    X, y = prepare_training_data(image_dir, edge_method, target_size, sample_limit)
    
    if len(X) == 0:
        print("No training data prepared. Check image directory.")
        return None
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create model
    model = create_edge_detection_model(input_shape=X_train[0].shape)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=5, monitor='val_loss')
    ]
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Load the best model
    model = load_model(model_path)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model

def predict_edges(model, image_path, target_size=(256, 256)):
    """
    Predict edges in an image using the trained model
    
    Parameters:
    model (tf.keras.Model): Trained model
    image_path (str): Path to the input image
    target_size (tuple): Target size for input image
    
    Returns:
    tuple: (original_image, predicted_edges)
    """
    # Preprocess image
    img = preprocess_image(image_path, target_size)
    
    if img is None:
        return None, None
    
    # Predict edges
    pred = model.predict(img)
    
    # Reshape predictions to remove batch dimension
    pred = pred[0, :, :, 0]
    
    # Denormalize and convert to uint8 for displaying
    pred_uint8 = (pred * 255).astype(np.uint8)
    
    # Load original image for visualization
    original = cv2.imread(image_path)
    original = cv2.resize(original, target_size)
    
    return original, pred_uint8

def compare_edge_detection_methods(image_path, cnn_model=None, target_size=(256, 256)):
    """
    Compare different edge detection methods
    
    Parameters:
    image_path (str): Path to the input image
    cnn_model (tf.keras.Model): Trained CNN model (optional)
    target_size (tuple): Target size for input image
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Generate edge maps using traditional methods
    canny_edges = generate_edge_map(img, method='canny')
    sobel_edges = generate_edge_map(img, method='sobel')
    laplacian_edges = generate_edge_map(img, method='laplacian')
    
    # Convert to uint8 for display
    canny_edges_uint8 = (canny_edges * 255).astype(np.uint8)
    sobel_edges_uint8 = (sobel_edges * 255).astype(np.uint8)
    laplacian_edges_uint8 = (laplacian_edges * 255).astype(np.uint8)
    
    # Prepare figure
    if cnn_model is not None:
        # Preprocess image for CNN
        img_norm = img.astype('float32') / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)
        
        # Predict edges using CNN
        cnn_edges = cnn_model.predict(img_batch)[0, :, :, 0]
        cnn_edges_uint8 = (cnn_edges * 255).astype(np.uint8)
        
        # Plot with CNN prediction
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("Canny Edge Detection")
        plt.imshow(canny_edges_uint8, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("Sobel Edge Detection")
        plt.imshow(sobel_edges_uint8, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title("Laplacian Edge Detection")
        plt.imshow(laplacian_edges_uint8, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("CNN Edge Detection")
        plt.imshow(cnn_edges_uint8, cmap='gray')
        plt.axis('off')
    else:
        # Plot without CNN prediction
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title("Canny Edge Detection")
        plt.imshow(canny_edges_uint8, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title("Sobel Edge Detection")
        plt.imshow(sobel_edges_uint8, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title("Laplacian Edge Detection")
        plt.imshow(laplacian_edges_uint8, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_detection_comparison.png')
    plt.show()

def edge_detection_demo():
    """
    Demo using a pre-trained model or traditional methods
    """
    # Model path
    model_path = "edge_detection_model.h5"
    
    # Check if a trained model exists
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model = load_model(model_path)
    else:
        print("No pre-trained model found. Using traditional methods only.")
        model = None
    
    while True:
        print("\nEdge Detection Demo")
        print("1. Detect edges in an image")
        print("2. Compare edge detection methods")
        print("3. Train a new model")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            image_path = input("Enter the path to the image: ")
            
            if model:
                # Use CNN model
                original, edges = predict_edges(model, image_path)
                
                if original is not None and edges is not None:
                    # Display results
                    plt.figure(figsize=(10, 5))
                    
                    plt.subplot(1, 2, 1)
                    plt.title("Original Image")
                    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.title("CNN Edge Detection")
                    plt.imshow(edges, cmap='gray')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
            else:
                # Use traditional methods
                img = cv2.imread(image_path)
                if img is not None:
                    # Generate and display edges using Canny
                    edges = generate_edge_map(img, method='canny')
                    edges_uint8 = (edges * 255).astype(np.uint8)
                    
                    plt.figure(figsize=(10, 5))
                    
                    plt.subplot(1, 2, 1)
                    plt.title("Original Image")
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.title("Canny Edge Detection")
                    plt.imshow(edges_uint8, cmap='gray')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"Error: Could not read image from {image_path}")
        
        elif choice == '2':
            image_path = input("Enter the path to the image: ")
            compare_edge_detection_methods(image_path, model)
        
        elif choice == '3':
            image_dir = input("Enter the directory containing training images: ")
            sample_limit_str = input("Enter the maximum number of samples to use (or press Enter for all): ")
            
            sample_limit = None
            if sample_limit_str.strip():
                try:
                    sample_limit = int(sample_limit_str)
                except ValueError:
                    print("Invalid input. Using all available samples.")
            
            epochs = 10
            epochs_str = input(f"Enter the number of epochs (default: {epochs}): ")
            if epochs_str.strip():
                try:
                    epochs = int(epochs_str)
                except ValueError:
                    print(f"Invalid input. Using default: {epochs} epochs.")
            
            print("Training model...")
            model = train_edge_detection_model(
                image_dir, 
                model_path=model_path,
                epochs=epochs,
                sample_limit=sample_limit
            )
            
            if model is not None:
                print(f"Model trained and saved to {model_path}")
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

def main():
    edge_detection_demo()

if __name__ == "__main__":
    main()
