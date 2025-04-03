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
    
    # Get image dimensions
    height, width, channels = img.shape
    print(f"Image Dimensions: {width} x {height}")
    
    # Display original image
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    # 1. Rotation
    # Define the center point (center of the image)
    center = (width // 2, height // 2)
    
    # Define the rotation angle (in degrees)
    # You can change this angle as needed
    angle = 45
    
    # Define the scale factor
    scale = 1.0
    
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply the rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    rotated_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    
    # Display rotated image
    plt.subplot(2, 3, 2)
    plt.title(f"Rotated ({angle} degrees)")
    plt.imshow(rotated_rgb)
    plt.axis('off')
    
    # 2. Scaling
    # Define the scaling factors for width and height
    # You can change these factors as needed
    scale_factor_x = 1.5  # Increase width by 50%
    scale_factor_y = 1.5  # Increase height by 50%
    
    # Calculate new dimensions
    new_width = int(width * scale_factor_x)
    new_height = int(height * scale_factor_y)
    
    # Apply scaling (resize)
    # INTER_LINEAR is a bilinear interpolation
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # For display, resize back to original size to fit in the subplot
    display_scaled = cv2.resize(scaled_img, (width, height), interpolation=cv2.INTER_LINEAR)
    display_scaled_rgb = cv2.cvtColor(display_scaled, cv2.COLOR_BGR2RGB)
    
    # Display scaled image
    plt.subplot(2, 3, 3)
    plt.title(f"Scaled ({scale_factor_x}x, {scale_factor_y}x)")
    plt.imshow(display_scaled_rgb)
    plt.axis('off')
    
    # 3. Downscaling
    # Define the scaling factors for width and height
    # You can change these factors as needed
    downscale_factor_x = 0.5  # Reduce width by 50%
    downscale_factor_y = 0.5  # Reduce height by 50%
    
    # Calculate new dimensions
    new_width_down = int(width * downscale_factor_x)
    new_height_down = int(height * downscale_factor_y)
    
    # Apply downscaling (resize)
    downscaled_img = cv2.resize(img, (new_width_down, new_height_down), interpolation=cv2.INTER_AREA)
    
    # For display, resize back to original size to fit in the subplot
    display_downscaled = cv2.resize(downscaled_img, (width, height), interpolation=cv2.INTER_LINEAR)
    display_downscaled_rgb = cv2.cvtColor(display_downscaled, cv2.COLOR_BGR2RGB)
    
    # Display downscaled image
    plt.subplot(2, 3, 4)
    plt.title(f"Downscaled ({downscale_factor_x}x, {downscale_factor_y}x)")
    plt.imshow(display_downscaled_rgb)
    plt.axis('off')
    
    # 4. Translation
    # Define the translation values (in pixels)
    # You can change these values as needed
    tx = 100  # Positive for right, negative for left
    ty = 50   # Positive for down, negative for up
    
    # Create the translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    translated_img = cv2.warpAffine(img, translation_matrix, (width, height))
    translated_rgb = cv2.cvtColor(translated_img, cv2.COLOR_BGR2RGB)
    
    # Display translated image
    plt.subplot(2, 3, 5)
    plt.title(f"Translated ({tx}px, {ty}px)")
    plt.imshow(translated_rgb)
    plt.axis('off')
    
    # 5. Combined transformation: Rotation + Scaling + Translation
    # First, create a rotation matrix with additional scaling
    combined_angle = 30
    combined_scale = 0.8
    rotation_scale_matrix = cv2.getRotationMatrix2D(center, combined_angle, combined_scale)
    
    # Add translation to the matrix
    rotation_scale_matrix[0, 2] += 50  # x translation
    rotation_scale_matrix[1, 2] += 20  # y translation
    
    # Apply the combined transformation
    combined_img = cv2.warpAffine(img, rotation_scale_matrix, (width, height))
    combined_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
    
    # Display combined transformation
    plt.subplot(2, 3, 6)
    plt.title("Combined Transformation")
    plt.imshow(combined_rgb)
    plt.axis('off')
    
    # Save transformed images (optional)
    cv2.imwrite("rotated_image.jpg", rotated_img)
    cv2.imwrite("scaled_image.jpg", scaled_img)
    cv2.imwrite("downscaled_image.jpg", downscaled_img)
    cv2.imwrite("translated_image.jpg", translated_img)
    cv2.imwrite("combined_transformation.jpg", combined_img)
    
    # Show all plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
