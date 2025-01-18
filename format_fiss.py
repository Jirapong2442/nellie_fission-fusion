import numpy as np
import pandas as pd
import os

# shift event from the first frame of fission
# 
path = "./nellie_output/mdivi/"
out_path = "./nellie_output/glu/"
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
    if "fission" in file and os.path.isfile(os.path.join(path,file)):
        output_fission.append(os.path.join(path,file))


for file in output_fission:
    filename = file
    df = pd.read_csv(file) 
    df = np.array(df)
    rolled_df = np.roll(df, -1)
    rolled_df = pd.DataFrame(rolled_df)
    if len(file) == 59:
        rolled_df.to_csv(os.path.join(out_path,file[-37:]),   index=False)

    elif len(file) == 55:
        rolled_df.to_csv(os.path.join(out_path,file[-33:]),   index=False)

    elif len(file) == 64:
        rolled_df.to_csv(os.path.join(out_path,file[-42:]),   index=False)

    elif len(file) == 58:
        rolled_df.to_csv(os.path.join(out_path,file[-36:]),   index=False)
    

        

