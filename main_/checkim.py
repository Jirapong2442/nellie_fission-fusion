'''
This script visualizes labeled images and their corresponding centroids using Napari.
It reads a CSV file containing feature information, a segmented image file, and a reassigned label image file which found in Nellie's output.
'''
import tifffile
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import center_of_mass, extrema
import napari 
from skimage import measure

def plot_labels(csvPath,segmented_path,reassigned_path):
    
    nellie_df_2d = pd.read_csv(csvPath)
    labeled_im = tifffile.imread(segmented_path)
    reassigned_im = tifffile.imread(reassigned_path)
    frames = labeled_im.shape[0]

    viewer = napari.Viewer()
    viewer.add_labels(reassigned_im, name ="labeled image")
   
    
    centroids = []
    labels = []
    #contour_points = []
    for frame in range(frames):
        labeled_im_frame = labeled_im[frame]
        unique_label = np.unique(labeled_im_frame)
        unique_label = unique_label[unique_label != 0]
        num_features = len(unique_label) 

        print(f"feature in {frame:02} frame is {num_features}")

        # !!!!!cannot find a label since it always start over if we use range(len(unique_label))
        for label in unique_label:
            all_extrema = center_of_mass(labeled_im_frame == label)
            centroid = all_extrema#[3]
            centroid = np.array(list(centroid))
            centroids.append(np.append(frame, centroid))
            
            nellie_df_2d_label = nellie_df_2d[nellie_df_2d['label'] == label]
            reassigned_label = nellie_df_2d_label['reassigned_label_raw']
            labels.append(int(list(reassigned_label)[0]))
        #add number of assigned label
            #if label == contour_label:
             #       contours = get_label_contour(labeled_im_frame, label)
              #      for contour in contours:
               #         contour_points.append(np.column_stack((np.full(len(contour), frame), contour)))
    label_strings = [str(l) for l in labels]
    text = {
        'string': label_strings,  # Displays each label (e.g., '1', '2', etc.)
        'size': 0.1,           # Text size (adjust for visibility; avoid very small values like 2)
        'color': 'green',     # Text color (string or RGB array)
        'translation': np.array([-10, 10])  # Optional: Offset text from point (x, y in pixels)
    }
    viewer.add_points(np.array(centroids),  text = labels, size = 0.5, name = "label number")
    napari.run()
def find_extrema(binary_image):
    # Ensure the image is binary
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8) * 255
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Get the largest contour
    cnt = max(contours, key=cv2.contourArea)
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    # Calculate extrema points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    # Calculate additional points
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_right = (x + w, y + h)
    bottom_left = (x, y + h)
    # Combine all points
    extrema = np.array([
    top_left,
    topmost,
    top_right,
    rightmost,
    bottom_right,
    bottommost,
    bottom_left,
    leftmost
    ])
    
    return extrema



main_dir  = "D:/Internship/NTU/simulation/hi_fission/"
#test the result from iminstance
seg_path = main_dir + "hi_fiss.ome-TYX-T1p0_Y0p25_X0p25-ch0-t0_to_300-im_instance_label.ome.tif" #don
reassigned_path = main_dir +"hi_fiss.ome-TYX-T1p0_Y0p25_X0p25-ch0-t0_to_300-im_obj_label_reassigned.ome.tif"
CSV_file_path = main_dir + "hi_fiss.ome-TYX-T1p0_Y0p25_X0p25-ch0-t0_to_300-features_organelles.csv"

# Usage
labeled_im = tifffile.imread(seg_path)
labeled_im_frame = labeled_im[0]
binary_image = labeled_im_frame == 602 # your binary image

# requirement 
# 1. feature component csv file
# 2. instance label ome tif 
# 3. obk lable reassigned ome tif file
plot_labels(CSV_file_path,seg_path,reassigned_path)
