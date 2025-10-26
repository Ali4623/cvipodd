import cv2
import numpy as np

# 1. Load and display the original and grayscale image
# --- Make sure to replace 'your_image_path.jpg' with the actual path to your image ---
image = cv2.imread('S:\photo.png') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray Scale', gray)
cv2.waitKey(0)

# ---------------------------------
# 2. Sobel Edge Detection
# ---------------------------------
# Apply Sobel operator in x and y directions
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Convert back to 8-bit unsigned integers
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combine the x and y Sobel images
sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

# Display Sobel results
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Sobel Combined', sobel_combined)
cv2.waitKey(0)

# ---------------------------------
# 3. Canny Edge Detection
# ---------------------------------
# Apply Canny edge detector
canny_edges = cv2.Canny(gray, 100, 200)

cv2.imshow('Canny Edge Detection', canny_edges)
cv2.waitKey(0)

# ---------------------------------
# 4. Thresholding on Grayscale
# ---------------------------------
# Apply simple binary thresholding
ret, binary_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('Binary Threshold', binary_thresh)
cv2.waitKey(0)

# ---------------------------------
# 5. Thresholding on Color
# ---------------------------------
# Split the color image into its B, G, R channels
b, g, r = cv2.split(image)

# Apply thresholding to each channel
ret_b, thresh_b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
ret_g, thresh_g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
ret_r, thresh_r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)

# Merge the thresholded channels back together
color_thresh = cv2.merge((thresh_b, thresh_g, thresh_r))

cv2.imshow('Color Threshold Image', color_thresh)
cv2.waitKey(0)

# Clean up all windows
cv2.destroyAllWindows()

