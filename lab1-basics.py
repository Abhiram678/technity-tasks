import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Input image path
    # You can replace this with the path to your image file
    image_path = "input_image.jpg"
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display original image
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    # Get image dimensions and information
    height, width, channels = img.shape
    print(f"Image Dimensions: {width} x {height}")
    print(f"Number of Channels: {channels}")
    
    # Basic image manipulations
    
    # 1. Cropping
    # Crop parameters (adjust as needed based on your image)
    # Format: img[y_start:y_end, x_start:x_end]
    start_y, end_y = int(height * 0.25), int(height * 0.75)
    start_x, end_x = int(width * 0.25), int(width * 0.75)
    
    # Crop the image
    cropped_img = img[start_y:end_y, start_x:end_x]
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    
    # Display cropped image
    plt.subplot(2, 3, 2)
    plt.title("Cropped Image")
    plt.imshow(cropped_rgb)
    plt.axis('off')
    
    # 2. Horizontal Flip
    flipped_horizontal = cv2.flip(img, 1)  # 1 = horizontal flip
    flipped_horizontal_rgb = cv2.cvtColor(flipped_horizontal, cv2.COLOR_BGR2RGB)
    
    # Display horizontally flipped image
    plt.subplot(2, 3, 3)
    plt.title("Horizontal Flip")
    plt.imshow(flipped_horizontal_rgb)
    plt.axis('off')
    
    # 3. Vertical Flip
    flipped_vertical = cv2.flip(img, 0)  # 0 = vertical flip
    flipped_vertical_rgb = cv2.cvtColor(flipped_vertical, cv2.COLOR_BGR2RGB)
    
    # Display vertically flipped image
    plt.subplot(2, 3, 4)
    plt.title("Vertical Flip")
    plt.imshow(flipped_vertical_rgb)
    plt.axis('off')
    
    # 4. Both horizontal and vertical flip
    flipped_both = cv2.flip(img, -1)  # -1 = both horizontal and vertical
    flipped_both_rgb = cv2.cvtColor(flipped_both, cv2.COLOR_BGR2RGB)
    
    # Display both flipped image
    plt.subplot(2, 3, 5)
    plt.title("Both Flips")
    plt.imshow(flipped_both_rgb)
    plt.axis('off')
    
    # Save modified images (optional)
    cv2.imwrite("cropped_image.jpg", cropped_img)
    cv2.imwrite("horizontal_flip.jpg", flipped_horizontal)
    cv2.imwrite("vertical_flip.jpg", flipped_vertical)
    cv2.imwrite("both_flips.jpg", flipped_both)
    
    # Show all plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
