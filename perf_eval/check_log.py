'''
plot linechart comparision of simulated fission fusion event (log) and predicted fission fusion event (area-baed algorithm output)
    X axis = frame
    Y axis = number of event
'''

import numpy as np
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
from checkFissFus2 import get_fiss_fus_all 
from plot_area_label import plot_multiple_line
from significance_test import sliding_window_cumsum
import matplotlib.pyplot as plt
from  main_.checkim import plot_labels

# query .log data
event_all_log = []

with open("D:/Internship/NTU/my_script/simulated_output/thick_log/mito_high_fi_nonzero_frames.log", "r", encoding='utf-8-sig') as f:
    string = f.read()
    lines = string.split("\n")
    for line in lines:
        if len(line ) != 0:
            events = line.split(":")[1].split(",")
            frame = line.split(":")[0].split()[1]
           
            if len(events) == 1 :
                continue
            else:
                num_events = [x.split("=")[1] for x in events]
                x= int(num_events[0]) + float(num_events[1])
                num = [float(frame),x,float(num_events[2])]
                event_all_log.append(num)
# event_all_log = [frame ,fusion , fission]

#event log = frame that has either fission or fusion
event_all_log = (np.transpose(np.array(event_all_log)))
event_all_log[1] = np.where(event_all_log[1] == 0, event_all_log[1] + 0.01, event_all_log[1])

mask = event_all_log[0] > 50 # fram 50 or above
test_fission = event_all_log[2][mask]
test_fusion = event_all_log[1][mask]
test_frame = (event_all_log[0][mask] - 51).astype(int) # adjust it because we strat processing at frame 50 (50 become 0 )

fission_fusion_test = test_fission/test_fusion
test_event = event_all_log[1,0:201]
test_event  = np.array([test_fusion,test_fission])

#query fission fusion data
path_hifi = "D:/Internship/NTU/my_script/nellie_output/simulated/thick_fission/shifted"
fission_hifi, fusion_hifi, fiss_frame, fus_frame = get_fiss_fus_all(path_hifi)
#fission_hifi = sliding_window_cumsum(fission_hifi,)

adjusted_fusion =np.where(fusion_hifi == 0, fusion_hifi+0.01,fusion_hifi)
#all_event= np.concatenate((fusion_hifi[:,1:],fission_hifi[:,1:],fission_hifi[:,1:]/ adjusted_fusion[:,1:]))
all_event= np.concatenate((np.expand_dims(fusion_hifi[0][test_frame],axis = 0),np.expand_dims(fission_hifi[0][test_frame],axis = 0),np.expand_dims(fission_hifi[0][test_frame]/ adjusted_fusion[0][test_frame], axis = 0)))

np.set_printoptions(suppress=True)
x = all_event[1,:] #predicted
y = test_event[1] #ground truth

point_y = np.array([x,y])
point_x = np.array([np.arange(len(x)), np.arange(len(x))])
res = pearsonr(x, y)

plot_multiple_line(point_x,point_y,point_x,["algo" , "ground truth"], title="fission fusion comparison")
# plotting the data
plt.scatter(np.arange(len(x)), x )
 
# This will fit the best line into the graph
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))
 #        (np.unique(x)), color='red')

plt.show()