import numpy as np
import pandas as pd
import os

'''
change to all post event because the simulated data annotate event at the end 
    fission: 0 pre -> 0 + 0 post: 
        - annotate post-event where we have neighbor (my algorithm)
        - the simulated data also annotate post-event      

    fusion: 0 + 0 -> 0
        - annotate pre-event where we have neighbor (my algorithm)
        - the simulated data also annotate post event

if we aim to compare the result of simulated algorithm and estimated event, need to shift the result of the algorithm output on fusion



'''
path = "./nellie_output/simulated/thick_fission/"
out_path = "./nellie_output/simulated/thick_fission/shifted/"
output_fission = []
output_fusion = []
fission_df = []
fusion_df = []

def get_fiss_fus (file_array , array):
    all_frame = []
    for file in file_array:
        df = pd.read_csv(file) 
        df_event = np.count_nonzero(df, axis=0).tolist()
        frame = np.expand_dims(np.arange(len(df_event)),axis=1)
        all_frame.append(frame)
        if len(array) == 0:
            array = [df_event]
        else:
            array.append(df_event)
    return array,all_frame


for file in os.listdir(path):
    if "fusion" in file and os.path.isfile(os.path.join(path,file)):
        output_fusion.append(os.path.join(path,file))


for file in output_fusion:
    filename = file
    df = pd.read_csv(file) 
    df = np.array(df)
    rolled_df = np.roll(df, 1)
    rolled_df = pd.DataFrame(rolled_df)
    rolled_df.to_csv(os.path.join(out_path,file.split('/')[-1]),   index=False)
 
    

        

