import cv2
import numpy as np
import os
import time

def main():
    # Input video path
    # You can replace this with the path to your video file with faces
    # If you want to use webcam, set video_path to 0
    video_path = "faces_video.mp4"  # or 0 for webcam
    
    # Output video path (optional)
    output_path = "face_eye_detection_output.mp4"
    
    # Paths to the Haar Cascade classifiers
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
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process and display each face
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Get the region of interest (ROI) for the face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
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
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                processed_fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {processed_fps:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display the result
            cv2.imshow('Face and Eye Detection', frame)
            
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

if __name__ == "__main__":
    main()
