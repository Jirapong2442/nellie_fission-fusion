import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

dir_path = "/home/jirapong/nellie/my_script/gridsearch/"
folders = os.listdir(dir_path)
result = []
name = []

for folder in folders:
    files = os.listdir(os.path.join(dir_path,folder))
    for file in files:
        if "diff" in file:
            value = np.load(os.path.join(dir_path,folder,file))
            value = np.sum(value,axis = 0)
            if len(result) == 0:
                result.append(value)
                result = np.array(result)
                name.append(file)
                name = np.array(name)
            else:
                result = np.vstack((result,value))
                name = np.vstack((name,file))

# TP TN FP FN
# more combination threshold = more positive = approching 11.
# less == approacing  00 
tpr = result[:,0]/(result[:,0] + result[:,3]) #TP/(TP+FN)
fpr = result[:,2]/(result[:,2] + result[:,1]) #TP/(TP+FN)

# Create the plot
# Create a simple scatter plot
'''
plt.figure(figsize=(10, 6))
plt.scatter(fpr, tpr, color='blue', s=100)

# Basic formatting
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Simple Scatter Plot')
plt.grid(True, alpha=0.3)

# Display the plot
plt.show()
'''
coords = np.array([
    [1, 2, 4, 6, 8],    # x coordinates
    [2, 5, 3, 8, 4]     # y coordinates
])
x_coords = coords[0]
y_coords = coords[1]

# Create a simple scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_coords, y_coords, color='blue', s=100)

# Basic formatting
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Simple Scatter Plot')
plt.grid(True, alpha=0.3)

# Display the plot
plt.show()
