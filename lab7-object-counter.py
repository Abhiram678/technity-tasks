import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict

def count_objects_contours(image_path, min_area=500, max_area=50000, display=True):
    """
    Count objects in an image using contour detection
    
    Parameters:
    image_path (str): Path to the input image
    min_area (int): Minimum contour area to consider
    max_area (int): Maximum contour area to consider
    display (bool): Whether to display the results
    
    Returns:
    int: Number of objects detected
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
    
    # Create a copy of the image to draw contours
    img_with_contours = img.copy()
    
    # Draw contours and number them
    for i, contour in enumerate(valid_contours):
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # If moment is zero, use the first point of the contour
            cX, cY = contour[0][0]
        
        # Draw contour and put number
        cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 2)
        cv2.putText(img_with_contours, str(i+1), (cX, cY), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display results if requested
    if display:
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Thresholded image
        plt.subplot(1, 3, 2)
        plt.title("Thresholded Image")
        plt.imshow(thresh, cmap='gray')
        plt.axis('off')
        
        # Image with contours
        plt.subplot(1, 3, 3)
        plt.title(f"Detected Objects: {len(valid_contours)}")
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return len(valid_contours)

def count_objects_blob(image_path, min_area=500, max_area=50000, display=True):
    """
    Count objects in an image using SimpleBlobDetector
    
    Parameters:
    image_path (str): Path to the input image
    min_area (int): Minimum blob area to consider
    max_area (int): Maximum blob area to consider
    display (bool): Whether to display the results
    
    Returns:
    int: Number of objects detected
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Set up the detector with parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    
    # Filter by circularity
    params.filterByCircularity = False
    
    # Filter by convexity
    params.filterByConvexity = False
    
    # Filter by inertia
    params.filterByInertia = False
    
    # Create detector with parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(gray)
    
    # Draw detected blobs
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), 
                                          (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Number the blobs
    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        cv2.putText(img_with_keypoints, str(i+1), (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display results if requested
    if display:
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Image with keypoints
        plt.subplot(1, 2, 2)
        plt.title(f"Detected Objects: {len(keypoints)}")
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return len(keypoints)

def count_colored_objects(image_path, color_ranges=None, display=True):
    """
    Count objects based on color ranges
    
    Parameters:
    image_path (str): Path to the input image
    color_ranges (dict): Dictionary of color ranges in HSV format
                        {color_name: ((h_min, s_min, v_min), (h_max, s_max, v_max))}
    display (bool): Whether to display the results
    
    Returns:
    dict: Dictionary with counts for each color
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return {}
    
    # Default color ranges if none provided
    if color_ranges is None:
        color_ranges = {
            'red': ((0, 100, 100), (10, 255, 255)),  # First range for red (wraps around)
            'red2': ((160, 100, 100), (180, 255, 255)),  # Second range for red
            'green': ((35, 50, 50), (85, 255, 255)),
            'blue': ((90, 50, 50), (130, 255, 255)),
            'yellow': ((20, 100, 100), (35, 255, 255))
        }
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create a dictionary to store masks for each color
    masks = {}
    
    # Create a dictionary to store contours for each color
    contours_dict = {}
    
    # Process each color
    for color_name, (lower, upper) in color_ranges.items():
        # Convert bounds to numpy arrays
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create a mask for the color
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Store the mask
        masks[color_name] = mask
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store the contours
        contours_dict[color_name] = contours
    
    # Special handling for red (which wraps around in HSV)
    if 'red' in masks and 'red2' in masks:
        # Combine the two red masks
        combined_red_mask = cv2.bitwise_or(masks['red'], masks['red2'])
        masks['red'] = combined_red_mask
        
        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_dict['red'] = contours
        
        # Remove the second red entry
        if 'red2' in contours_dict:
            del contours_dict['red2']
            del masks['red2']
    
    # Create a copy of the image to draw contours
    img_with_contours = img.copy()
    
    # Count objects for each color
    color_counts = {}
    for color_name, contours in contours_dict.items():
        # Skip red2 as it's combined with red
        if color_name == 'red2':
            continue
        
        # Create a color-specific display image
        color_img = np.zeros_like(img)
        
        # Count valid contours (filter by area if needed)
        valid_contours = []
        min_area = 100  # Minimum area to consider
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                valid_contours.append(contour)
                
                # Draw the contour on the color-specific image
                cv2.drawContours(color_img, [contour], -1, (0, 255, 0), 2)
                
                # Draw the contour on the main image
                if color_name == 'red':
                    contour_color = (0, 0, 255)  # Red (BGR)
                elif color_name == 'green':
                    contour_color = (0, 255, 0)  # Green
                elif color_name == 'blue':
                    contour_color = (255, 0, 0)  # Blue
                elif color_name == 'yellow':
                    contour_color = (0, 255, 255)  # Yellow
                else:
                    contour_color = (255, 255, 255)  # White
                
                cv2.drawContours(img_with_contours, [contour], -1, contour_color, 2)
                
                # Get the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # If moment is zero, use the first point of the contour
                    cX, cY = contour[0][0]
                
                # Add the color name and object number
                cv2.putText(img_with_contours, f"{color_name[0].upper()}{len(valid_contours)}", 
                            (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Store the count
        color_counts[color_name] = len(valid_contours)
        
    # Combine all the individual masks to create a total mask
    total_mask = np.zeros_like(masks[list(masks.keys())[0]])
    for mask in masks.values():
        total_mask = cv2.bitwise_or(total_mask, mask)
    
    # Display results if requested
    if display:
        # Determine the number of rows needed for display
        n_colors = len(masks)
        n_rows = int(np.ceil((n_colors + 2) / 3))  # +2 for original and result
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        # Original image
        plt.subplot(n_rows, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Combined mask
        plt.subplot(n_rows, 3, 2)
        plt.title("Combined Mask")
        plt.imshow(total_mask, cmap='gray')
        plt.axis('off')
        
        # Result with contours
        plt.subplot(n_rows, 3, 3)
        plt.title("Detected Objects")
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Individual color masks
        for i, (color_name, mask) in enumerate(masks.items()):
            plt.subplot(n_rows, 3, i + 4)  # Start from the 4th position
            plt.title(f"{color_name.capitalize()} Mask (Count: {color_counts.get(color_name, 0)})")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return color_counts

def count_objects_video(video_path=0, method='contours', color_ranges=None, output_path=None):
    """
    Count objects in a video stream
    
    Parameters:
    video_path (str or int): Path to video file or camera index
    method (str): Method to use ('contours', 'blob', or 'color')
    color_ranges (dict): Dictionary of color ranges (only used if method is 'color')
    output_path (str): Path to save output video (optional)
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream or file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Default color ranges if method is 'color' and no ranges provided
    if method == 'color' and color_ranges is None:
        color_ranges = {
            'red': ((0, 100, 100), (10, 255, 255)),
            'red2': ((160, 100, 100), (180, 255, 255)),
            'green': ((35, 50, 50), (85, 255, 255)),
            'blue': ((90, 50, 50), (130, 255, 255)),
            'yellow': ((20, 100, 100), (35, 255, 255))
        }
    
    # Create blob detector if method is 'blob'
    if method == 'blob':
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 50000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
    
    # Start time for FPS calculation
    start_time = time.time()
    frame_count = 0
    
    # Initialize a dictionary to track object counts over time
    object_counts = defaultdict(list)
    timestamp_points = []
    
    # Process video
    while True:
        # Read frame
        ret, frame = cap.read()
        
        # Break loop if frame not read successfully
        if not ret:
            break
        
        # Increment frame counter
        frame_count += 1
        current_time = time.time() - start_time
        
        # Store timestamp for plotting
        if frame_count % 5 == 0:  # Store every 5th frame to reduce data points
            timestamp_points.append(current_time)
        
        # Process frame based on method
        if method == 'contours':
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = 500
            max_area = 50000
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    valid_contours.append(contour)
            
            # Draw contours and number them
            for i, contour in enumerate(valid_contours):
                # Get the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # If moment is zero, use the first point
                    cX, cY = contour[0][0]
                
                # Draw contour and put number
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.putText(frame, str(i+1), (cX, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Store count
            if frame_count % 5 == 0:
                object_counts['objects'].append(len(valid_contours))
            
            # Display count
            cv2.putText(frame, f"Objects: {len(valid_contours)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        elif method == 'blob':
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect blobs
            keypoints = detector.detect(gray)
            
            # Draw keypoints
            frame = cv2.drawKeypoints(frame, keypoints, np.array([]),
                                       (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Number the blobs
            for i, keypoint in enumerate(keypoints):
                x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
                cv2.putText(frame, str(i+1), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Store count
            if frame_count % 5 == 0:
                object_counts['objects'].append(len(keypoints))
            
            # Display count
            cv2.putText(frame, f"Objects: {len(keypoints)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        elif method == 'color':
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Process each color
            color_counts = {}
            for color_name, (lower, upper) in color_ranges.items():
                # Skip red2 as it's handled separately
                if color_name == 'red2':
                    continue
                
                # Convert bounds to numpy arrays
                lower = np.array(lower)
                upper = np.array(upper)
                
                # Create a mask for the color
                mask = cv2.inRange(hsv, lower, upper)
                
                # Special handling for red (which wraps around in HSV)
                if color_name == 'red' and 'red2' in color_ranges:
                    # Get the second red range
                    lower2 = np.array(color_ranges['red2'][0])
                    upper2 = np.array(color_ranges['red2'][1])
                    
                    # Create a mask for the second red range
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    
                    # Combine the two red masks
                    mask = cv2.bitwise_or(mask, mask2)
                
                # Apply morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area
                min_area = 100
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_area:
                        valid_contours.append(contour)
                
                # Set contour color based on color name
                if color_name == 'red':
                    contour_color = (0, 0, 255)  # BGR
                elif color_name == 'green':
                    contour_color = (0, 255, 0)
                elif color_name == 'blue':
                    contour_color = (255, 0, 0)
                elif color_name == 'yellow':
                    contour_color = (0, 255, 255)
                else:
                    contour_color = (255, 255, 255)
                
                # Draw contours and label
                for i, contour in enumerate(valid_contours):
                    # Draw contour
                    cv2.drawContours(frame, [contour], -1, contour_color, 2)
                    
                    # Get the center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = contour[0][0]
                    
                    # Add label
                    cv2.putText(frame, f"{color_name[0].upper()}{i+1}", (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Store count
                color_counts[color_name] = len(valid_contours)
                
                # Store counts over time
                if frame_count % 5 == 0:
                    object_counts[color_name].append(len(valid_contours))
            
            # Display counts
            y_pos = 30
            for color_name, count in color_counts.items():
                if color_name == 'red':
                    text_color = (0, 0, 255)
                elif color_name == 'green':
                    text_color = (0, 255, 0)
                elif color_name == 'blue':
                    text_color = (255, 0, 0)
                elif color_name == 'yellow':
                    text_color = (0, 255, 255)
                else:
                    text_color = (255, 255, 255)
                
                cv2.putText(frame, f"{color_name.capitalize()}: {count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                y_pos += 30
        
        # Calculate and display FPS
        fps_text = f"FPS: {frame_count / (time.time() - start_time):.1f}"
        cv2.putText(frame, fps_text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Write frame to output video if specified
        if output_path:
            out.write(frame)
        
        # Display frame
        cv2.imshow('Object Counter', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # Plot object counts over time if we have data
    if timestamp_points and any(object_counts.values()):
        plt.figure(figsize=(12, 6))
        
        for obj_type, counts in object_counts.items():
            if counts:  # Check if we have data for this type
                # Trim to match the number of timestamp points
                counts = counts[:len(timestamp_points)]
                plt.plot(timestamp_points, counts, label=obj_type.capitalize())
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Object Count')
        plt.title('Object Counts Over Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('object_count_over_time.png')
        plt.show()

def main():
    print("Object Counter")
    print("1. Count objects in image using contours")
    print("2. Count objects in image using blob detection")
    print("3. Count colored objects in image")
    print("4. Count objects in video/webcam")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        image_path = input("Enter the path to the image: ")
        min_area = int(input("Enter the minimum contour area (default 500): ") or 500)
        max_area = int(input("Enter the maximum contour area (default 50000): ") or 50000)
        
        count = count_objects_contours(image_path, min_area, max_area)
        print(f"Detected {count} objects")
    
    elif choice == '2':
        image_path = input("Enter the path to the image: ")
        min_area = int(input("Enter the minimum blob area (default 500): ") or 500)
        max_area = int(input("Enter the maximum blob area (default 50000): ") or 50000)
        
        count = count_objects_blob(image_path, min_area, max_area)
        print(f"Detected {count} objects")
    
    elif choice == '3':
        image_path = input("Enter the path to the image: ")
        
        print("Using default color ranges (red, green, blue, yellow)")
        color_counts = count_colored_objects(image_path)
        
        print("Object counts by color:")
        for color, count in color_counts.items():
            print(f"  {color.capitalize()}: {count}")
    
    elif choice == '4':
        print("1. Use webcam")
        print("2. Use video file")
        source_choice = input("Enter your choice (1-2): ")
        
        if source_choice == '1':
            video_path = 0  # Use default camera
        else:
            video_path = input("Enter the path to the video file: ")
        
        print("1. Count objects using contours")
        print("2. Count objects using blob detection")
        print("3. Count colored objects")
        method_choice = input("Enter your choice (1-3): ")
        
        if method_choice == '1':
            method = 'contours'
        elif method_choice == '2':
            method = 'blob'
        else:
            method = 'color'
        
        save_output = input("Save output video? (y/n): ").lower() == 'y'
        output_path = None
        if save_output:
            output_path = input("Enter the output path (default: output.mp4): ") or "output.mp4"
        
        count_objects_video(video_path, method, output_path=output_path)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
