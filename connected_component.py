
import numpy as np
import pandas as pd
from scipy import ndimage  

#generate binary structure function test (generally generate 4 connectivity in 2D)
file_path = "/home/jirapong/Mito_data/nellie_output/0804201min_timeseries_test.ome-ch0-features_branches.csv"
footprint = ndimage.generate_binary_structure(3, 1)
print(footprint)

#label function test
default_array = np.array([[0, 0, 1, 0, 0],
                          [1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0],
                          [1, 1, 1, 0, 0]])
connectivity = np.array([[1,1,1],
                         [1,1,1,],
                         [1,1,1,]])
# Labeling with 4-connectivity
labeled_array_4, num_labels_4 = ndimage.label(default_array)
labeled_array_8, num_labels_8 = ndimage.label(default_array,structure=connectivity)


# Printing labeled arrays and number of labels for 4-connectivity
print("Labeled array with 4-connectivity:")
print(labeled_array_4)
print("Number of labels (4-connectivity):", num_labels_4)

# Printing labeled arrays and number of labels for 8-connectivity
print("\nLabeled array with 8-connectivity:")
print(labeled_array_8)
print("Number of labels (8-connectivity):", num_labels_8)
