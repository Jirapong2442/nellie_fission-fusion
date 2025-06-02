import scipy.ndimage as ndi
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from matplotlib import pyplot as plt

path = 'D:/Internship/NTU/data_for_script/high_fission_binary.ome-TYX-T1p0_Y1p0_X1p0-ch0-t0_to_666-im_instance_label.ome.tif'
im_in = imread(path)

result = -ndi.gaussian_laplace(im_in[333], sigma=0.33) 
result_adjust = np.where(result == 0, 0.001, result)

max_filt = ndi.maximum_filter(result, size = ((9,9)))

peaks = (result_adjust == max_filt)

threshold = np.mean(peaks)  # You can adjust this threshold as needed
background_mask = peaks < threshold
#peaks[background_mask] = 0

coords = np.max(peaks, axis=0)
coords_idx = np.argwhere(coords)


fig = plt.figure(figsize=(500,500 ))
plt.subplot(2,2,1)
plt.imshow(result, cmap= 'gray')  # Display the original image slice
plt.title("result")

plt.subplot(2,2,2)
plt.imshow(max_filt, cmap= 'gray') 
plt.title("max_filt")

plt.subplot(2,2,3)
plt.imshow(peaks, cmap= 'gray') 
plt.title("peaks")

plt.subplot(2,2,4)
plt.imshow(background_mask, cmap= 'gray') 
plt.title("background_mask")
# Overlay the coordinates on the image
#plt.scatter(coords_idx[:, 1], coords_idx[:, 0], color='red', s=10)  # s is the size of the points

# Show the plot
plt.show()
