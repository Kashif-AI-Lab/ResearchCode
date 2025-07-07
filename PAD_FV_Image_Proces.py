
import matplotlib.pyplot as plt
import cv2


# Load the input image as grayscale
input_image = cv2.imread('E:/Gdnask University of Technology/Finger Vein Presentation Attack Detection System Using Deep Learning Model/IDIAPCROPPED/FAKE/001_L_1.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if input_image is None:
    raise Exception("Error: Unable to load the input image.")

# Apply a Gaussian filter for noise removal
blurred_image = cv2.GaussianBlur(input_image, (5, 5), 0)

# Display the blurred image
plt.figure(figsize=(8, 6))
plt.imshow(blurred_image,cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

# Perform contrast enhancement using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(blurred_image)

plt.imshow(enhanced_image,cmap='gray')
plt.title('Enhanced_image')
plt.axis('off')

# Display the original and enhanced images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Enhanced Image')
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Optionally, save the enhanced image
cv2.imwrite('enhanced_image.jpg', enhanced_image)