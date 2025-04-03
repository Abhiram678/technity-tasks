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
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Detected {len(cars)} Cars and {len(pedestrians)} Pedestrians")
    plt.imshow(img_with_detection_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the result (optional)
    cv2.imwrite("car_pedestrian_detection_result.jpg", img_with_detection)
    
    print("Detection complete.")

def detect_from_video():
    # Input video path
    # You can replace this with the path to your video file with cars and pedestrians
    # If you want to use webcam, set video_path to 0
    video_path = "street_video.mp4"  # or 0 for webcam
    
    # Output video path (optional)
    output_path = "car_pedestrian_detection_output.mp4"
    
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
    
    # Open the video
    if isinstance(video_path, int):
        cap = cv2.VideoCapture(video_path)
        print(f"Opening webcam")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Opening video: {video_path}")
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Create video writer (optional)
    save_output = True
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing loop
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            
            # If frame is not read correctly, break
            if not ret:
                print("End of video or error reading frame")
                break
            
            # Increment frame counter
            frame_count += 1
            
            # Process every 2nd frame to improve performance (optional)
            if frame_count % 2 != 0 and frame_count > 1:
                if save_output:
                    out.write(frame)
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect cars
            cars = car_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            # Draw rectangles around cars
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Car', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Detect pedestrians
            pedestrians = pedestrian_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 80)
            )
            
            # Draw rectangles around pedestrians
            for (x, y, w, h) in pedestrians:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display detection counts
            cv2.putText(frame, f"Cars: {len(cars)}, Pedestrians: {len(pedestrians)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                processed_fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {processed_fps:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Display the result
            cv2.imshow('Car and Pedestrian Detection', frame)
            
            # Write the frame to output video (if enabled)
            if save_output:
                out.write(frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted")
                break
    
    except Exception as e:
        print(f"Error during video processing: {e}")
    
    finally:
        # Release resources
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        if frame_count > 0:
            print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
            print(f"Average FPS: {frame_count / elapsed_time:.2f}")
        
        if save_output:
            print(f"Output video saved to: {output_path}")

def main():
    print("1. Detect cars and pedestrians from image")
    print("2. Detect cars and pedestrians from video")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        detect_from_image()
    elif choice == '2':
        detect_from_video()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
