import tifffile
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import center_of_mass, extrema
import napari 
import matplotlib.pyplot as plt
from skimage import measure

with open("D:/Internship/NTU/simulation/pdb_simulation/test_fission_position.log", "r", encoding='utf-8-sig') as f:
    string = f.read()
    lines = string.split("\n")
    num = 1
    event_num = []
    x_coordinates = []
    y_coordinates = []
    for line in lines:

        if len(line ) != 0:
            if (int(num) - 1)% 3  == 0: 
                event_num.append(num)
                num+=1
            
            else: 
                x = line[1:-1].split(',')[0]
                y = line[1:-1].split(',')[1]
                x_coordinates.append(float(x))
                y_coordinates.append(float(y))
                num+=1

        
print(x_coordinates)