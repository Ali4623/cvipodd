import cv2

# Load the image
image = cv2.imread(r'C:\Users\phata\OneDrive\Desktop\5b.png')  # Replace with actual path
original = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and bounding boxes
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  # Filter out small objects
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(original, [cnt], -1, (0, 0, 255), 1)

# Show results
cv2.imshow('Contours with Bounding Boxes', original)
cv2.waitKey(0)
cv2.destroyAllWindows()
