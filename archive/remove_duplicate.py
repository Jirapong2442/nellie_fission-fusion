#check all possible area combination that could not make up to 
import numpy as np
import pandas as pd
import os

def get_label(frame, data, isFusion,):
    sub_event = data[data["Frame"] == frame]
    sub_event_masked = sub_event[sub_event["isFusion"] == isFusion]

    dup= sub_event_masked[sub_event_masked.duplicated(['Nearest Label'], keep=False)]

    dup_label = pd.unique(dup["Nearest Label"])
    return dup,dup_label

# folder path
dir_path = 'D:/Internship/NTU/algo_output/'

# list to store files
output_fission = []
output_fusion = []
event = []

# Iterate directory
for path in os.listdir(dir_path):
    if  "original" in path:
        for file in  os.listdir(os.path.join(dir_path, path)):
        # check if current path is a file
            if "output_fission" in file and os.path.isfile(os.path.join(dir_path, path,file)):
                output_fission.append(os.path.join(dir_path, path,file))
            if "output_fusion" in file and os.path.isfile(os.path.join(dir_path, path,file)):
                output_fusion.append(os.path.join(dir_path, path,file))
            if "event" in file and os.path.isfile(os.path.join(dir_path, path,file)):
                event.append(os.path.join(dir_path, path,file))
        
for index in range(len(output_fission)):
    #event_path = "D:/Internship/NTU/algo_output/Start_1st_frame/1_0_event.csv"
    #fiss = "D:/Internship/NTU/algo_output/Start_1st_frame/1_0_output_fission.csv"
    #fus = "D:/Internship/NTU/algo_output/Start_1st_frame/1_0_output_fusion.csv"
    event_path = event[index]
    fiss = output_fission[index]
    fus = output_fusion[index]

    df = pd.read_csv(event_path)
    final_fission_volume = np.array(pd.read_csv(fiss))
    final_fusion_volume = np.array(pd.read_csv(fus))
    for frame in pd.unique(df["Frame"]):
        dup_fission,label_fission = get_label(frame,df,isFusion=0)
        dup_fusion,label_fusion = get_label(frame,df,isFusion=1)
        frame_index = frame.astype(int)

        for label in label_fission: 
            to_remove = np.array(dup_fission[dup_fission['Nearest Label'] == label][1:]['Label'].values)
            for remove in to_remove:
                index = remove.astype(int)
                final_fission_volume[index-2][frame_index] = 0
            #final_fission_volume[list(map(int, (to_remove-2).tolist()))][:][10]  = 0
            
        for label in label_fusion: 
            to_remove = np.array(dup_fusion[dup_fusion['Nearest Label'] == label][1:]['Label'].values)
            for remove in to_remove:
                index = remove.astype(int)
                final_fusion_volume[index-2][frame_index+1] = 0
        np.savetxt("adjusted" + os.path.basename(fiss),final_fission_volume, delimiter=',', fmt='%f')
        np.savetxt("adjusted" + os.path.basename(fus),final_fusion_volume, delimiter=',', fmt='%f')


