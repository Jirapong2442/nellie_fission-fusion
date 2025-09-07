import tifffile
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import center_of_mass, extrema
import napari 
import matplotlib.pyplot as plt
from skimage import measure

with open("D:/Internship/NTU/simulation/nonzero_frames_final.log", "r", encoding='utf-8-sig') as f:
    event_all = []
    string = f.read()
    lines = string.split("\n")
    for line in lines:
        if len(line ) != 0:
            events = line.split(":")[1].split(",")
           
            if len(events) == 1 :
                continue
            else:
                num_events = [x.split("=")[1] for x in events]
                x= int(num_events[0]) + int(num_events[1])
                num = [x,int(num_events[2])]
                event_all.append(num)
#fusion , fission
event = pd.DataFrame(event_all, columns=['fusion', 'fission'])
print(np.sum(event, axis = 0))

        
