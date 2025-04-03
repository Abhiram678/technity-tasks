import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
import os
from imutils import face_utils

def align_face(image_path, output_path=None, display=True):
    """
    Align a face in an image based on eye positions
    
    Parameters:
    image_path (str): Path to the input image
    output_path (str): Path to save the aligned face (optional)
    display (bool): Whether to display the results
    
    Returns:
    numpy.ndarray: Aligned face image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize dlib's face detector and facial landmark predictor
    # You need to download the shape predictor file from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # Extract it and provide the path below
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.isfile(predictor_path):
        print(f"Error: Shape predictor file not found at {predictor_path}")
        print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return None
    
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Detect faces in the grayscale image
    faces = detector(gray, 1)
    
    # Check if any faces were detected
    if len(faces) == 0:
        print("No faces detected in the image")
        return None
    
    # We'll work with the first face
    face = faces[0]
    
    # Get facial landmarks
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)
    
    # The indices for the left and right eyes
    # In the 68-point facial landmark detector:
    # Points 36-41 represent the right eye
    # Points 42-47 represent the left eye
    left_eye_indices = list(range(42, 48))
    right_eye_indices = list(range(36, 42))
    
    # Calculate the center of each eye
    left_eye_center = landmarks[left_eye_indices].mean(axis=0).astype("int")
    right_eye_center = landmarks[right_eye_indices].mean(axis=0).astype("int")
    
    # Calculate the angle between the eye centers
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Calculate the desired position of the right eye
    # We want the right eye to be 45% from the left, and both eyes on the same y-coordinate
    desired_right_eye_x = 0.45
    desired_dist = desired_right_eye_x - (1.0 - desired_right_eye_x)
    
    # Calculate the scale based on the distance between the eyes
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = desired_dist * 256  # Desired face width
    scale = desired_dist / dist
    
    # Calculate the center point between the eyes
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                   (left_eye_center[1] + right_eye_center[1]) // 2)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    
    # Update the translation component of the matrix
    tX = 256 * 0.5
    tY = 256 * 0.4
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    
    # Apply the affine transformation
    aligned_face = cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_CUBIC)
    
    # Save the aligned face if output_path is provided
    if output_path:
        cv2.imwrite(output_path, aligned_face)
        print(f"Aligned face saved to {output_path}")
    
    # Display the results if requested
    if display:
        # Create a copy of the image to draw on
        img_with_landmarks = img.copy()
        
        # Draw circles at each eye landmark
        for (x, y) in landmarks[left_eye_indices]:
            cv2.circle(img_with_landmarks, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in landmarks[right_eye_indices]:
            cv2.circle(img_with_landmarks, (x, y), 2, (0, 255, 0), -1)
        
        # Draw circles at the eye centers
        cv2.circle(img_with_landmarks, tuple(left_eye_center), 4, (0, 0, 255), -1)
        cv2.circle(img_with_landmarks, tuple(right_eye_center), 4, (0, 0, 255), -1)
        
        # Draw a line between the eye centers
        cv2.line(img_with_landmarks, tuple(left_eye_center), tuple(right_eye_center), (255, 0, 0), 2)
        
        # Convert BGR to RGB for displaying with matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_with_landmarks_rgb = cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB)
        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        # Display the results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Detected Landmarks")
        plt.imshow(img_with_landmarks_rgb)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Aligned Face")
        plt.imshow(aligned_face_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return aligned_face

def align_multiple_faces(image_path, output_dir=None, display=True):
    """
    Align all faces in an image based on eye positions
    
    Parameters:
    image_path (str): Path to the input image
    output_dir (str): Directory to save aligned faces (optional)
    display (bool): Whether to display the results
    
    Returns:
    list: List of aligned face images
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize dlib's face detector and facial landmark predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.isfile(predictor_path):
        print(f"Error: Shape predictor file not found at {predictor_path}")
        print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return None
    
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Detect faces in the grayscale image
    faces = detector(gray, 1)
    
    # Check if any faces were detected
    if len(faces) == 0:
        print("No faces detected in the image")
        return None
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each face
    aligned_faces = []
    face_count = 0
    
    for face in faces:
        face_count += 1
        
        # Get facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        # The indices for the left and right eyes
        left_eye_indices = list(range(42, 48))
        right_eye_indices = list(range(36, 42))
        
        # Calculate the center of each eye
        left_eye_center = landmarks[left_eye_indices].mean(axis=0).astype("int")
        right_eye_center = landmarks[right_eye_indices].mean(axis=0).astype("int")
        
        # Calculate the angle between the eye centers
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Desired right eye position and scale calculation
        desired_right_eye_x = 0.45
        desired_dist = desired_right_eye_x - (1.0 - desired_right_eye_x)
        
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = desired_dist * 256
        scale = desired_dist / dist
        
        # Calculate the center point between the eyes
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update the translation component of the matrix
        tX = 256 * 0.5
        tY = 256 * 0.4
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply the affine transformation
        aligned_face = cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_CUBIC)
        aligned_faces.append(aligned_face)
        
        # Save the aligned face if output_dir is provided
        if output_dir:
            output_path = os.path.join(output_dir, f"aligned_face_{face_count}.jpg")
            cv2.imwrite(output_path, aligned_face)
            print(f"Aligned face {face_count} saved to {output_path}")
    
    # Display the results if requested
    if display and aligned_faces:
        # Create a copy of the image to draw on
        img_with_faces = img.copy()
        
        # Convert BGR to RGB for displaying with matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw rectangles around the faces
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        img_with_faces_rgb = cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB)
        
        # Calculate grid size for displaying aligned faces
        grid_size = int(np.ceil(np.sqrt(len(aligned_faces))))
        
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Display original image with detected faces
        plt.subplot(grid_size, grid_size, 1)
        plt.title("Detected Faces")
        plt.imshow(img_with_faces_rgb)
        plt.axis('off')
        
        # Display each aligned face
        for i, aligned_face in enumerate(aligned_faces):
            aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            plt.subplot(grid_size, grid_size, i + 2)
            plt.title(f"Aligned Face {i+1}")
            plt.imshow(aligned_face_rgb)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return aligned_faces

def align_face_from_webcam():
    """
    Capture video from webcam and align faces in real-time
    """
    # Initialize dlib's face detector and facial landmark predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.isfile(predictor_path):
        print(f"Error: Shape predictor file not found at {predictor_path}")
        print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit, 's' to save the current aligned face")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        # Create a copy to display
        display_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray, 0)
        
        # Process each face
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            # Draw rectangle around the face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # The indices for the left and right eyes
            left_eye_indices = list(range(42, 48))
            right_eye_indices = list(range(36, 42))
            
            # Draw landmarks for the eyes
            for (x, y) in landmarks[left_eye_indices]:
                cv2.circle(display_frame, (x, y), 2, (0, 0, 255), -1)
            for (x, y) in landmarks[right_eye_indices]:
                cv2.circle(display_frame, (x, y), 2, (0, 0, 255), -1)
            
            # Calculate the center of each eye
            left_eye_center = landmarks[left_eye_indices].mean(axis=0).astype("int")
            right_eye_center = landmarks[right_eye_indices].mean(axis=0).astype("int")
            
            # Draw eye centers
            cv2.circle(display_frame, tuple(left_eye_center), 4, (255, 0, 0), -1)
            cv2.circle(display_frame, tuple(right_eye_center), 4, (255, 0, 0), -1)
            
            # Draw line between eyes
            cv2.line(display_frame, tuple(left_eye_center), tuple(right_eye_center), (255, 255, 0), 2)
            
            # Calculate alignment parameters
            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Display the angle
            cv2.putText(display_frame, f"Angle: {angle:.1f} degrees", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the result
        cv2.imshow('Face Alignment (Press q to quit, s to save)', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # If 's' is pressed, save an aligned face
        if key == ord('s') and len(faces) > 0:
            # We'll align and save the first face
            face = faces[0]
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            # Calculate alignment parameters
            left_eye_indices = list(range(42, 48))
            right_eye_indices = list(range(36, 42))
            left_eye_center = landmarks[left_eye_indices].mean(axis=0).astype("int")
            right_eye_center = landmarks[right_eye_indices].mean(axis=0).astype("int")
            
            # Calculate the angle between the eye centers
            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Desired right eye position and scale calculation
            desired_right_eye_x = 0.45
            desired_dist = desired_right_eye_x - (1.0 - desired_right_eye_x)
            
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desired_dist = desired_dist * 256
            scale = desired_dist / dist
            
            # Calculate the center point between the eyes
            eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                           (left_eye_center[1] + right_eye_center[1]) // 2)
            
            # Get the rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
            
            # Update the translation component of the matrix
            tX = 256 * 0.5
            tY = 256 * 0.4
            M[0, 2] += (tX - eyes_center[0])
            M[1, 2] += (tY - eyes_center[1])
            
            # Apply the affine transformation
            aligned_face = cv2.warpAffine(frame, M, (256, 256), flags=cv2.INTER_CUBIC)
            
            # Save the aligned face
            output_path = f"aligned_face_webcam_{int(time.time())}.jpg"
            cv2.imwrite(output_path, aligned_face)
            print(f"Aligned face saved to {output_path}")
        
        # If 'q' is pressed, break the loop
        if key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    import time
    
    print("Choose an option:")
    print("1. Align face from image")
    print("2. Align multiple faces from image")
    print("3. Align faces from webcam")
    
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == '1':
        # Align a single face from an image
        image_path = input("Enter the path to the image: ")
        output_path = f"aligned_face_{int(time.time())}.jpg"
        align_face(image_path, output_path, display=True)
    
    elif choice == '2':
        # Align multiple faces from an image
        image_path = input("Enter the path to the image: ")
        output_dir = f"aligned_faces_{int(time.time())}"
        align_multiple_faces(image_path, output_dir, display=True)
    
    elif choice == '3':
        # Align faces from webcam
        align_face_from_webcam()
    
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
