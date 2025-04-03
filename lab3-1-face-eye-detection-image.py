import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Input image path
    # You can replace this with the path to your image file with faces
    image_path = "faces.jpg"
    
    # Paths to the Haar Cascade classifiers
    # You might need to adjust these paths based on your OpenCV installation
    # By default, these are usually in the OpenCV installation directory
    # e.g. /usr/local/share/opencv4/haarcascades/ on Linux
    # or C:/opencv/build/etc/haarcascades/ on Windows
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    
    # Check if cascade files exist
    if not os.path.isfile(face_cascade_path):
        print(f"Error: Face cascade file not found at {face_cascade_path}")
        return
    
    if not os.path.isfile(eye_cascade_path):
        print(f"Error: Eye cascade file not found at {eye_cascade_path}")
        return
    
    # Load the cascade classifiers
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale (required for Haar Cascade)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    # Parameters:
    # - scaleFactor: Parameter specifying how much the image size is reduced at each image scale
    # - minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
    # - minSize: Minimum possible object size. Objects smaller than this are ignored
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"Found {len(faces)} faces!")
    
    # Create a copy of the image to draw on
    img_with_detection = img.copy()
    
    # Draw rectangles around the faces and detect eyes
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img_with_detection, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Get the region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_with_detection[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(5, 5)
        )
        
        # Draw rectangles around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_with_detection_rgb = cv2.cvtColor(img_with_detection, cv2.COLOR_BGR2RGB)
    
    # Display the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Detected {len(faces)} Faces")
    plt.imshow(img_with_detection_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the result (optional)
    cv2.imwrite("face_eye_detection_result.jpg", img_with_detection)
    
    # Alternative method - DNN based face detection
    print("\nUsing DNN-based face detection:")
    
    # Paths to the DNN model files
    # You need to download these from:
    # https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
    # and
    # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    
    # Check if model files exist
    if os.path.isfile(model_file) and os.path.isfile(config_file):
        # Load the DNN model
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        
        # Create a copy of the image to draw on
        img_dnn_detection = img.copy()
        
        # Get image dimensions
        (h, w) = img.shape[:2]
        
        # Create a blob from the image
        # Parameters:
        # - image: input image
        # - scalefactor: scaling factor for the image
        # - size: spatial size for the output image
        # - mean: mean subtraction values
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network and get detections
        net.setInput(blob)
        detections = net.forward()
        
        # Count detected faces
        dnn_face_count = 0
        
        # Process each detection
        for i in range(0, detections.shape[2]):
            # Extract the confidence
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > 0.5:  # 50% confidence threshold
                dnn_face_count += 1
                
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw the bounding box of the face
                cv2.rectangle(img_dnn_detection, (startX, startY), (endX, endY), (0, 0, 255), 2)
                
                # Display the confidence
                text = f"{confidence * 100:.2f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(img_dnn_detection, text, (startX, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        print(f"DNN found {dnn_face_count} faces!")
        
        # Convert BGR to RGB for displaying with matplotlib
        img_dnn_detection_rgb = cv2.cvtColor(img_dnn_detection, cv2.COLOR_BGR2RGB)
        
        # Display the results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"DNN Detected {dnn_face_count} Faces")
        plt.imshow(img_dnn_detection_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the result (optional)
        cv2.imwrite("face_dnn_detection_result.jpg", img_dnn_detection)
    else:
        print("DNN model files not found. Skipping DNN-based detection.")
        if not os.path.isfile(model_file):
            print(f"Missing model file: {model_file}")
        if not os.path.isfile(config_file):
            print(f"Missing config file: {config_file}")
        print("Download instructions are in the comments.")

if __name__ == "__main__":
    main()
