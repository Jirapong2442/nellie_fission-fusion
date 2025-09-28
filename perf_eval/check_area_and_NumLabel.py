'''
for each condition, check number of label and area over time/frame
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# check number of label over time/frame 
# check area over time/frame
stream_frame = ['1','1','1','1','5','5','10','10']
stream_con = ['0','0.5','1','2','1','0.5','0','2']

for s in range(len(stream_con)):
    im_concen = stream_con[s]
    im_frame = stream_frame[s]
    print(im_concen+"_"+im_frame)

    main_dir = "/home/jirapong/jirapong/archrive/nellie_output/"
    out_path = "/home/jirapong/nellie/my_script/check_label&area"
    output_name = im_frame + "_" + im_concen + "area_label_check"

    file_path_feature = main_dir + "/" + im_frame + "_resized_" + im_concen + "_glu_1min_#1.ome-ch0-features_components.csv"

    df = pd.read_csv(file_path_feature)
    mini_df = df[['t', 'reassigned_label_raw' , 'area_raw'] ]

    frames = np.unique(mini_df['t'])
    label_num_all = []
    area_all = []
    for frame in frames:
        mini_df_frame = mini_df[mini_df['t'] == frame]
        label_num = len(np.unique(mini_df_frame['reassigned_label_raw']))
        area = np.sum(mini_df_frame['area_raw'])

        label_num_all.append(label_num)
        area_all.append(area)
    
    label_NN_all = pd.DataFrame({ 'label_num': label_num_all, 'area': area_all})
    label_NN_all.to_csv(os.path.join(out_path,f'{output_name}_neighbour.csv'), index=False) 


# fission/fusion over time/frame