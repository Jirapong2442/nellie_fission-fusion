import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.draw import polygon
from skimage.color import label2rgb
from scipy.ndimage import center_of_mass
from skimage.color import rgb2gray

# Create a binary image
binary_image = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 0, 0, 0],
                         [0, 1, 1, 0, 1, 1],
                         [0, 0, 0, 0, 1, 1]], dtype=np.uint8)

# Find contours
label_image = measure.label(binary_image, connectivity=binary_image.ndim)
contours = measure.find_contours(binary_image, level=0.5)

# Draw contours on a color image
contour_image = label2rgb(label_image, image=binary_image, bg_label=0)
grayscale = rgb2gray(contour_image)

# Plot the binary image and contours
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Contours')
plt.imshow(contour_image)
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
plt.axis('off')

for contour in contours:
    print(contour)
plt.show()

print()