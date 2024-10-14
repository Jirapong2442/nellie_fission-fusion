import tifffile
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import center_of_mass, extrema
import napari
from scipy.ndimage import center_of_mass   

import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from skimage import measure
'''
def get_label_contour(label_data, label_id):
    label_mask = label_data == label_id
    contours = measure.find_contours(label_mask, 0.5)
    return [np.flip(contour, axis=1) for contour in contours]

        def get_label_contour(label_data, label_id):
            label_mask = label_data == label_id
            contour = measure.find_contours(label_mask, 0.5)
            contours = []
            for i in range(len(contour)):
                contours.append(np.insert(contour[i], 0, 1, axis=1))
# all contour
            return [np.flip(contour, axis=1) for contour in contours]

'''
def plot_labels(csvPath,segmented_path,reassigned_path,contour_label):
    
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
            all_extrema = extrema(labeled_im_frame == label)
            centroid = all_extrema[3]
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

    viewer.add_points(centroids, text = labels, size =0.5, name = "label number")
    napari.run()

    '''
        # Get contours for specific labels (e.g., labels 1, 2, and 3)
        label_ids_to_highlight = [321]
        all_contours = []
        for label_id in label_ids_to_highlight:
            contour = get_label_contour(labeled_im_frame, label_id)
            all_contours.extend(contour)
    '''
        
    '''
    if contour_points:
        contour_layer = viewer.add_shapes(
            np.transpose(contour_points),
            shape_type='polygon',
            edge_width=2,
            edge_color='yellow',
            face_color='transparent',
            name=f'Contour for Label {contour_label}'
        )
    else:
        print(f"No contour found for label {contour_label}")
    '''
    


#check extremas
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




#test the result from iminstance
seg_path = "D:/Internship/NTU/nellie_output/start 5/5_resized_1_glu_1min_#1.ome-ch0-im_instance_label.ome.tif"
reassigned_path = "D:/Internship/NTU/nellie_output/start 5/5_resized_1_glu_1min_#1.ome-ch0-im_obj_label_reassigned.ome.tif"
CSV_file_path = "D:/Internship/NTU/nellie_output/start 5/5_resized_1_glu_1min_#1.ome-ch0-features_components.csv"

# Usage
labeled_im = tifffile.imread(seg_path)
labeled_im_frame = labeled_im[0]
binary_image = labeled_im_frame == 602 # your binary image

# requirement 
# 1. feature component csv file
# 2. instance label ome tif 
# 3. obk lable reassigned ome tif file
plot_labels(CSV_file_path,seg_path,reassigned_path,321)

extremas = find_extrema(binary_image)
centriod = center_of_mass(binary_image)


# Plot the binary image and contours
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.scatter(extremas[:, 1], extremas[:, 0], color='red', s=100, edgecolor='black')
plt.scatter(centriod[1], centriod[0], color='blue', s=100, edgecolor='black')
plt.axis('off')
plt.show()

print()






'''
viewer = napari.Viewer()
for frame in range(30):
    labeled_im_frame = labeled_im[frame]

    unique_label = np.unique(labeled_im_frame)
    unique_label = unique_label[unique_label != 0]
    num_features = len(unique_label) 

    print(f"feature in {frame:02} frame is {num_features}")

    centroids = []
    labels = []
    # !!!!!cannot find a label since it always start over if we use range(len(unique_label))
    for label in unique_label:
        all_extrema = extrema(labeled_im_frame == label)
        centroid = all_extrema[3]
        centroid = np.array(list(centroid))
        centroids.append(np.append(frame, centroid))
        
        nellie_df_2d_label = nellie_df_2d[nellie_df_2d['label'] == label]
        reassigned_label = nellie_df_2d_label['reassigned_label_raw']
        
        labels.append(int(list(reassigned_label)[0]))
    #add number of assigned label
    viewer.add_points(centroids, text = labels, size =2, name = "label number")

viewer.add_labels(reassigned_im, name ="labeled image")
napari.run()
'''