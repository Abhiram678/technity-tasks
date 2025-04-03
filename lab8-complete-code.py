import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import time
from collections import Counter

def extract_features(image_path, feature_detector='sift'):
    """
    Extract features from an image
    
    Parameters:
    image_path (str): Path to the input image
    feature_detector (str): Feature detector to use ('sift', 'orb', or 'brisk')
    
    Returns:
    numpy.ndarray: Extracted features
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector
    if feature_detector.lower() == 'sift':
        detector = cv2.SIFT_create()
    elif feature_detector.lower() == 'orb':
        detector = cv2.ORB_create()
    elif feature_detector.lower() == 'brisk':
        detector = cv2.BRISK_create()
    else:
        raise ValueError(f"Unknown feature detector: {feature_detector}")
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    # If no features detected, return empty array
    if descriptors is None:
        return np.array([])
    
    return descriptors

def create_bag_of_words(feature_list, k=100):
    """
    Create a bag of words model by clustering the features
    
    Parameters:
    feature_list (list): List of feature arrays
    k (int): Number of clusters (vocabulary size)
    
    Returns:
    sklearn.cluster.KMeans: Trained KMeans model
    """
    # Concatenate all features
    all_features = np.vstack(feature_list)
    
    print(f"Clustering {all_features.shape[0]} features into {k} clusters...")
    
    # Create and fit KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(all_features)
    
    return kmeans

def create_histogram(features, kmeans_model):
    """
    Create a histogram of visual words for an image
    
    Parameters:
    features (numpy.ndarray): Features extracted from an image
    kmeans_model (sklearn.cluster.KMeans): Trained KMeans model
    
    Returns:
    numpy.ndarray: Histogram of visual words
    """
    # If no features, return zeros
    if features.size == 0:
        return np.zeros(kmeans_model.n_clusters)
    
    # Predict cluster for each feature
    predictions = kmeans_model.predict(features)
    
    # Create histogram
    histogram = np.zeros(kmeans_model.n_clusters)
    for prediction in predictions:
        histogram[prediction] += 1
    
    # Normalize histogram
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    
    return histogram

def prepare_dataset(dataset_path, feature_detector='sift', k=100, test_size=0.2):
    """
    Prepare a dataset for Bag of Words image classification
    
    Parameters:
    dataset_path (str): Path to the dataset directory
    feature_detector (str): Feature detector to use
    k (int): Vocabulary size
    test_size (float): Proportion of data to use for testing
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, kmeans_model, class_names)
    """
    # Get list of class directories
    class_dirs = [d for d in glob.glob(os.path.join(dataset_path, "*")) if os.path.isdir(d)]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {dataset_path}")
    
    # Get class names
    class_names = [os.path.basename(d) for d in class_dirs]
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create mapping from class name to label
    class_to_label = {class_name: i for i, class_name in enumerate(class_names)}
    
    # Initialize lists for images and labels
    image_paths = []
    labels = []
    
    # Collect all image paths and labels
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        class_label = class_to_label[class_name]
        
        # Get all images in this class
        class_images = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                      glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                      glob.glob(os.path.join(class_dir, "*.png"))
        
        # Add to lists
        image_paths.extend(class_images)
        labels.extend([class_label] * len(class_images))
    
    # Extract features from all images
    print(f"Extracting features from {len(image_paths)} images...")
    all_features = []
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}", end='\r')
        features = extract_features(image_path, feature_detector)
        if features is not None and features.size > 0:
            all_features.append(features)
        else:
            print(f"\nWarning: No features extracted from {image_path}")
    print()  # Print newline after progress updates
    
    # Create bag of words model
    kmeans_model = create_bag_of_words(all_features, k)
    
    # Create histograms for all images
    print("Creating histograms...")
    X = []
    y = []
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        print(f"Processing histogram {i+1}/{len(image_paths)}", end='\r')
        features = extract_features(image_path, feature_detector)
        if features is not None and features.size > 0:
            histogram = create_histogram(features, kmeans_model)
            X.append(histogram)
            y.append(label)
    print()  # Print newline after progress updates
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, kmeans_model, class_names

def train_svm_classifier(X_train, y_train, kernel='rbf'):
    """
    Train an SVM classifier
    
    Parameters:
    X_train (numpy.ndarray): Training data
    y_train (numpy.ndarray): Training labels
    kernel (str): Kernel type for SVM
    
    Returns:
    sklearn.svm.SVC: Trained SVM model
    """
    # Normalize/scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid for grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    # Create and train SVM classifier
    svm = SVC(kernel=kernel, probability=True, random_state=42)
    
    print("Training SVM classifier with grid search...")
    
    # Perform grid search
    grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Create a model with the best parameters
    best_svm = SVC(kernel=kernel, C=grid_search.best_params_['C'], 
                  gamma=grid_search.best_params_['gamma'],
                  probability=True, random_state=42)
    
    # Train on full training set
    best_svm.fit(X_train_scaled, y_train)
    
    return best_svm, scaler

def evaluate_classifier(model, scaler, X_test, y_test, class_names):
    """
    Evaluate the trained classifier
    
    Parameters:
    model (sklearn.svm.SVC): Trained model
    scaler (sklearn.preprocessing.StandardScaler): Trained scaler
    X_test (numpy.ndarray): Test data
    y_test (numpy.ndarray): Test labels
    class_names (list): List of class names
    
    Returns:
    float: Accuracy score
    """
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy

def save_model(kmeans_model, svm_model, scaler, class_names, feature_detector, output_dir='model'):
    """
    Save the trained models
    
    Parameters:
    kmeans_model (sklearn.cluster.KMeans): Trained KMeans model
    svm_model (sklearn.svm.SVC): Trained SVM model
    scaler (sklearn.preprocessing.StandardScaler): Trained scaler
    class_names (list): List of class names
    feature_detector (str): Feature detector used
    output_dir (str): Directory to save models
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save KMeans model
    with open(os.path.join(output_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans_model, f)
    
    # Save SVM model
    with open(os.path.join(output_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)
    
    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save class names
    with open(os.path.join(output_dir, 'class_names.pkl'), 'wb') as f:
        pickle.dump(class_names, f)
    
    # Save feature detector name
    with open(os.path.join(output_dir, 'feature_detector.txt'), 'w') as f:
        f.write(feature_detector)
    
    print(f"Models saved to {output_dir}")

def load_model(model_dir='model'):
    """
    Load saved models
    
    Parameters:
    model_dir (str): Directory containing saved models
    
    Returns:
    tuple: (kmeans_model, svm_model, scaler, class_names, feature_detector)
    """
    # Load KMeans model
    with open(os.path.join(model_dir, 'kmeans_model.pkl'), 'rb') as f:
        kmeans_model = pickle.load(f)
    
    # Load SVM model
    with open(os.path.join(model_dir, 'svm_model.pkl'), 'rb') as f:
        svm_model = pickle.load(f)
    
    # Load scaler
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load class names
    with open(os.path.join(model_dir, 'class_names.pkl'), 'rb') as f:
        class_names = pickle.load(f)
    
    # Load feature detector name
    with open(os.path.join(model_dir, 'feature_detector.txt'), 'r') as f:
        feature_detector = f.read().strip()
    
    return kmeans_model, svm_model, scaler, class_names, feature_detector

def classify_image(image_path, kmeans_model, svm_model, scaler, class_names, feature_detector):
    """
    Classify a single image
    
    Parameters:
    image_path (str): Path to the image
    kmeans_model (sklearn.cluster.KMeans): Trained KMeans model
    svm_model (sklearn.svm.SVC): Trained SVM model
    scaler (sklearn.preprocessing.StandardScaler): Trained scaler
    class_names (list): List of class names
    feature_detector (str): Feature detector to use
    
    Returns:
    tuple: (predicted_class, probability)
    """
    # Extract features
    features = extract_features(image_path, feature_detector)
    
    # Check if features were extracted
    if features is None or features.size == 0:
        print(f"Warning: No features extracted from {image_path}")
        return None, 0.0
    
    # Create histogram
    histogram = create_histogram(features, kmeans_model)
    
    # Scale histogram
    histogram_scaled = scaler.transform([histogram])
    
    # Predict class
    prediction = svm_model.predict(histogram_scaled)[0]
    probabilities = svm_model.predict_proba(histogram_scaled)[0]
    
    # Get predicted class name and probability
    predicted_class = class_names[prediction]
    probability = probabilities[prediction]
    
    return predicted_class, probability

def visualize_features(image_path, feature_detector='sift'):
    """
    Visualize features detected in an image
    
    Parameters:
    image_path (str): Path to the image
    feature_detector (str): Feature detector to use
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector
    if feature_detector.lower() == 'sift':
        detector = cv2.SIFT_create()
    elif feature_detector.lower() == 'orb':
        detector = cv2.ORB_create()
    elif feature_detector.lower() == 'brisk':
        detector = cv2.BRISK_create()
    else:
        raise ValueError(f"Unknown feature detector: {feature_detector}")
    
    # Detect keypoints
    keypoints = detector.detect(gray, None)
    
    # Draw keypoints
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display original and keypoints images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"{feature_detector.upper()} Features")
    plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_visualization.png')
    plt.show()
    
    print(f"Detected {len(keypoints)} keypoints")

def compare_feature_detectors(image_path):
    """
    Compare different feature detectors on an image
    
    Parameters:
    image_path (str): Path to the image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detectors
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()
    brisk = cv2.BRISK_create()
    
    # Detect keypoints
    start_time = time.time()
    sift_keypoints = sift.detect(gray, None)
    sift_time = time.time() - start_time
    
    start_time = time.time()
    orb_keypoints = orb.detect(gray, None)
    orb_time = time.time() - start_time
    
    start_time = time.time()
    brisk_keypoints = brisk.detect(gray, None)
    brisk_time = time.time() - start_time
    
    # Draw keypoints
    sift_img = cv2.drawKeypoints(img, sift_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    orb_img = cv2.drawKeypoints(img, orb_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    brisk_img = cv2.drawKeypoints(img, brisk_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title(f"SIFT Features ({len(sift_keypoints)} keypoints, {sift_time:.3f}s)")
    plt.imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title(f"ORB Features ({len(orb_keypoints)} keypoints, {orb_time:.3f}s)")
    plt.imshow(cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title(f"BRISK Features ({len(brisk_keypoints)} keypoints, {brisk_time:.3f}s)")
    plt.imshow(cv2.cvtColor(brisk_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_detector_comparison.png')
    plt.show()
    
    # Print comparison
    print(f"SIFT: {len(sift_keypoints)} keypoints in {sift_time:.3f}s")
    print(f"ORB: {len(orb_keypoints)} keypoints in {orb_time:.3f}s")
    print(f"BRISK: {len(brisk_keypoints)} keypoints in {brisk_time:.3f}s")

def main():
    print("Bag of Words Image Classification")
    print("1. Train a new model")
    print("2. Test model on images")
    print("3. Visualize features")
    print("4. Compare feature detectors")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        # Train a new model
        dataset_path = input("Enter the path to the dataset directory: ")
        
        # Choose feature detector
        print("Select feature detector:")
        print("1. SIFT (better but slower)")
        print("2. ORB (faster but less accurate)")
        print("3. BRISK (balanced)")
        detector_choice = input("Enter your choice (1-3): ")
        
        if detector_choice == '1':
            feature_detector = 'sift'
        elif detector_choice == '2':
            feature_detector = 'orb'
        elif detector_choice == '3':
            feature_detector = 'brisk'
        else:
            print("Invalid choice. Using SIFT as default.")
            feature_detector = 'sift'
        
        # Get vocabulary size
        k = int(input("Enter vocabulary size (default 100): ") or 100)
        
        # Prepare dataset
        X_train, X_test, y_train, y_test, kmeans_model, class_names = prepare_dataset(
            dataset_path, feature_detector, k)
        
        # Train SVM classifier
        svm_model, scaler = train_svm_classifier(X_train, y_train)
        
        # Evaluate classifier
        evaluate_classifier(svm_model, scaler, X_test, y_test, class_names)
        
        # Save models
        save_model(kmeans_model, svm_model, scaler, class_names, feature_detector)
    
    elif choice == '2':
        # Test model on images
        model_dir = input("Enter the model directory (default 'model'): ") or 'model'
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            print(f"Error: Model directory {model_dir} does not exist")
            return
        
        # Load models
        try:
            kmeans_model, svm_model, scaler, class_names, feature_detector = load_model(model_dir)
            print(f"Loaded models from {model_dir}")
            print(f"Feature detector: {feature_detector}")
            print(f"Classes: {class_names}")
        except Exception as e:
            print(f"Error loading models: {e}")
            return
        
        while True:
            # Get image path
            image_path = input("Enter the path to an image (or 'q' to quit): ")
            
            if image_path.lower() == 'q':
                break
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"Error: File {image_path} does not exist")
                continue
            
            # Classify image
            predicted_class, probability = classify_image(
                image_path, kmeans_model, svm_model, scaler, class_names, feature_detector)
            
            if predicted_class is None:
                continue
            
            # Display result
            print(f"Predicted class: {predicted_class} (probability: {probability:.4f})")
            
            # Show image with result
            img = cv2.imread(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Predicted: {predicted_class} ({probability:.4f})")
            plt.axis('off')
            plt.show()
    
    elif choice == '3':
        # Visualize features
        image_path = input("Enter the path to an image: ")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} does not exist")
            return
        
        # Choose feature detector
        print("Select feature detector:")
        print("1. SIFT")
        print("2. ORB")
        print("3. BRISK")
        detector_choice = input("Enter your choice (1-3): ")
        
        if detector_choice == '1':
            feature_detector = 'sift'
        elif detector_choice == '2':
            feature_detector = 'orb'
        elif detector_choice == '3':
            feature_detector = 'brisk'
        else:
            print("Invalid choice. Using SIFT as default.")
            feature_detector = 'sift'
        
        # Visualize features
        visualize_features(image_path, feature_detector)
    
    elif choice == '4':
        # Compare feature detectors
        image_path = input("Enter the path to an image: ")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} does not exist")
            return
        
        # Compare feature detectors
        compare_feature_detectors(image_path)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
