import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def detect_from_image():
    # Input image path
    # You can replace this with the path to your image file with cars and pedestrians
    image_path = "street_scene.jpg"
    
    # Paths to the Haar Cascade classifiers
    car_cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
    pedestrian_cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
    
    # Check if cascade files exist
    if not os.path.isfile(car_cascade_path):
        print(f"Error: Car cascade file not found at {car_cascade_path}")
        return
    
    if not os.path.isfile(pedestrian_cascade_path):
        print(f"Error: Pedestrian cascade file not found at {pedestrian_cascade_path}")
        return
    
    # Load the cascade classifiers
    car_cascade = cv2.CascadeClassifier(car_cascade_path)
    pedestrian_cascade = cv2.CascadeClassifier(pedestrian_cascade_path)
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Create a copy of the image to draw on
    img_with_detection = img.copy()
    
    # Convert to grayscale (required for Haar Cascade)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect cars
    cars = car_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    print(f"Found {len(cars)} cars!")
    
    # Draw rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(img_with_detection, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img_with_detection, 'Car', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Detect pedestrians
    pedestrians = pedestrian_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 80)
    )
    
    print(f"Found {len(pedestrians)} pedestrians!")
    
    # Draw rectangles around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(img_with_detection, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_with_detection, 'Person', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_with_detection_rgb = cv2.cvtColor(img_with_detection, cv2.COLOR_BGR2RGB)
    
    # Display the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title