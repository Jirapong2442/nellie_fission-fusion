import scipy.ndimage
from analysis import get_nellie_inputs, get_nellie_outputs
import tifffile
import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt


def check_fission_fusion_MM(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        row = [0 if x.lower() == 'nan' else int(x) for x in line.split(',')]
        data.append(row)
    array = np.array(data)
    event_number = np.count_nonzero(array,axis = 0)
    return event_number

def plot_corr(x,y,corr_score):
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x,y, color='blue', alpha=0.7)

    # Add correlation line
    z = np.polyfit(x,y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r-", alpha=0.7)

    # Customize the plot
    plt.title(f'Correlation Plot (r = {corr_score[0]:.2f}, p = {corr_score[1]:.4f})')
    plt.xlabel('events from first frame')
    plt.ylabel('events from tenth frame')

    # Add text box with correlation info
    text_box = f'Correlation: {corr_score[0]:.2f}\np-value: {corr_score[1]:.4f}'
    plt.text(0.05, 0.95, text_box, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Show the plot
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#reassigned_path = "/home/jirapong/Mito_data/nellie_output/test.ome-ch0-im_obj_label_reassigned.ome.tif"
#file_path = "/home/jirapong/Mito_data/nellie_output/test.ome-ch0-features_components.csv"
#seg_path = "/home/jirapong/Mito_data/nellie_output/test.ome-ch0-im_instance_label.ome.tif"

first_frame_fission_path_05 = "D:/Internship/NTU/algo_output/Start_1st_frame/1_0_output_fission.csv"
first_frame_fusion_path_05 = "D:/Internship/NTU/algo_output/Start_1st_frame/1_0_output_fusion.csv"
tenth_frame_fission_path_05 = "D:/Internship/NTU/algo_output/Start_10th_frame/10_0_output_fission.csv"
tenth_frame_fusion_path_05 = "D:/Internship/NTU/algo_output/Start_10th_frame/10_0_output_fusion.csv"


first_frame_fission_path_2 = "D:/Internship/NTU/algo_output/Start_1st_frame/1_2_output_fission.csv"
first_frame_fusion_path_2 = "D:/Internship/NTU/algo_output/Start_1st_frame/1_2_output_fusion.csv"
tenth_frame_fission_path_2 = "D:/Internship/NTU/algo_output/Start_10th_frame/10_2_output_fission.csv"
tenth_frame_fusion_path_2 = "D:/Internship/NTU/algo_output/Start_10th_frame/10_2_output_fusion.csv"


#read file
first_frame_fission_05 = pd.read_csv(first_frame_fission_path_05)
first_frame_fusion_05 = pd.read_csv(first_frame_fusion_path_05)
tenth_frame_fission_05 = pd.read_csv(tenth_frame_fission_path_05)
tenth_frame_fusion_05 = pd.read_csv(tenth_frame_fusion_path_05)


first_frame_fission_2 = pd.read_csv(first_frame_fission_path_2)
first_frame_fusion_2 = pd.read_csv(first_frame_fusion_path_2)
tenth_frame_fission_2 = pd.read_csv(tenth_frame_fission_path_2)
tenth_frame_fusion_2 = pd.read_csv(tenth_frame_fusion_path_2)

# count non zero
first_frame_fission_0_event = np.count_nonzero(first_frame_fission_05, axis=0)
first_frame_fusion_0_event = np.count_nonzero(first_frame_fusion_05, axis=0)
tenth_frame_fission_0_event = np.count_nonzero(tenth_frame_fission_05, axis=0)
tenth_frame_fusion_0_event = np.count_nonzero(tenth_frame_fusion_05, axis=0)

first_frame_fission_2_event = np.count_nonzero(first_frame_fission_2, axis=0)
first_frame_fusion_2_event = np.count_nonzero(first_frame_fusion_2, axis=0)
tenth_frame_fission_2_event = np.count_nonzero(tenth_frame_fission_2, axis=0)
tenth_frame_fusion_2_event = np.count_nonzero(tenth_frame_fusion_2, axis=0)


corr_fission_0 = scipy.stats.pearsonr(first_frame_fission_0_event[10:],tenth_frame_fission_0_event[1:])
corr_fusion_0 = scipy.stats.pearsonr(first_frame_fusion_0_event[10:],tenth_frame_fusion_0_event[1:])

corr_fission_2 = scipy.stats.pearsonr(first_frame_fission_2_event[10:],tenth_frame_fission_2_event[1:])
corr_fusion_2 = scipy.stats.pearsonr(first_frame_fusion_2_event[10:],tenth_frame_fusion_2_event[1:])


#plot_corr(first_frame_fission_0_event[10:],tenth_frame_fission_0_event[1:],corr_fission_0)
#plot_corr(first_frame_fission_0_event[10:],tenth_frame_fusion_0_event[1:],corr_fusion_0)

#plot_corr(first_frame_fission_2_event[10:],tenth_frame_fission_2_event[1:],corr_fission_2)
plot_corr(first_frame_fusion_2_event[10:],tenth_frame_fusion_2_event[1:],corr_fusion_2)
#plot_corr(tenth_frame_fission_05_event[1:],fifth_frame_fission_05_event[7:],corr_fission_05_3)

#plot_corr(first_frame_fusion_05_event[4:],fifth_frame_fusion_05_event[1:],corr_fusion_05_1)
#plot_corr(first_frame_fusion_05_event[10:],tenth_frame_fusion_05_event[1:],corr_fusion_05_2)
#plot_corr(tenth_frame_fusion_05_event[1:],fifth_frame_fusion_05_event[7:],corr_fusion_05_3)


corr_fusion_05 = np.sum(all_fusion_05_MM,all_fusion_05)
corr_fission_0 = np.sum(all_fission_0_MM,all_fission_0 )
corr_fusion_0= np.sum(all_fusion_0_MM,all_fusion_0)
corr_fission_1= np.sum(all_fission_1_MM, all_fission_1)
corr_fusion_1= np.sum(all_fusion_1_MM, all_fusion_1)
corr_fission_2= np.sum(all_fission_2_MM, all_fission_2)
corr_fusion_2= np.sum(all_fusion_2_MM, all_fusion_2)
corr_fission_control= np.sum(all_fission_control_MM, all_fission_control)
corr_fusion_control= np.sum(all_fusion_control_MM, all_fusion_control)

all_fission_05 = np.sum(fission_event_05)
all_fusion_05 = np.sum(fusion_event_05)
all_fission_0 = np.sum(fission_event_0)
all_fusion_0= np.sum(fusion_event_0)
all_fission_1= np.sum(fission_event_1)
all_fusion_1= np.sum(fusion_event_1)
all_fission_2= np.sum(fission_event_2)
all_fusion_2= np.sum(fusion_event_2)
all_fission_control= np.sum(fission_event_control)
all_fusion_control= np.sum(fusion_event_control)

'''
#from mitometer
fission_05 = "/home/jirapong/Mito_data/20240811_resized_0.5_glu_1min_#1.ome.tif_fission.txt"
fusion_05 = "/home/jirapong/Mito_data/20240811_resized_0.5_glu_1min_#1.ome.tif_fusion.txt"
fission_0 = "/home/jirapong/Mito_data/20240811_resized_0_glu_1min_#1.ome.tif_fission.txt"
fusion_0 = "/home/jirapong/Mito_data/20240811_resized_0_glu_1min_#1.ome.tif_fusion.txt"
fission_1 = "/home/jirapong/Mito_data/20240811_resized_1_glu_1min_#1.ome.tif_fission.txt"
fusion_1 = "/home/jirapong/Mito_data/20240811_resized_1_glu_1min_#1.ome.tif_fusion.txt"
fission_2 = "/home/jirapong/Mito_data/20240811_resized_2_glu_1min_#1.ome.tif_fission.txt"
fusion_2 = "/home/jirapong/Mito_data/20240811_resized_2_glu_1min_#1.ome.tif_fusion.txt"
fission_control = "/home/jirapong/Mito_data/20240811_resized_control_1min_#1.ome.tif_fission.txt"
fusion_control = "/home/jirapong/Mito_data/20240811_resized_control_1min_#1.ome.tif_fusion.txt"

fission_05_MM = check_fission_fusion_MM(fission_05)
fusion_05_MM = check_fission_fusion_MM(fusion_05)
fission_0_MM = check_fission_fusion_MM(fission_0)
fusion_0_MM = check_fission_fusion_MM(fusion_0)
fission_1_MM = check_fission_fusion_MM(fission_1)
fusion_1_MM = check_fission_fusion_MM(fusion_1)
fission_2_MM = check_fission_fusion_MM(fission_2)
fusion_2_MM = check_fission_fusion_MM(fusion_2)
fission_control_MM = check_fission_fusion_MM(fission_control)
fusion_control_MM = check_fission_fusion_MM(fusion_control)

all_fission_05_MM = np.sum(fission_05_MM)
all_fusion_05_MM = np.sum(fusion_05_MM)
all_fission_0_MM = np.sum(fission_0_MM)
all_fusion_0_MM = np.sum(fusion_0_MM)
all_fission_1_MM= np.sum(fission_1_MM)
all_fusion_1_MM= np.sum(fusion_1_MM)
all_fission_2_MM= np.sum(fission_2_MM)
all_fusion_2_MM= np.sum(fusion_2_MM)
all_fission_control_MM= np.sum(fission_control_MM)
all_fusion_control_MM= np.sum(fusion_control_MM)

#correlation of fission/fusion from my script and mito miter
corr_fission_05 = np.correlate(all_fission_05_MM,all_fission_05)
corr_fusion_05 = np.sum(all_fusion_05_MM,all_fusion_05)
corr_fission_0 = np.sum(all_fission_0_MM,all_fission_0 )
corr_fusion_0= np.sum(all_fusion_0_MM,all_fusion_0)
corr_fission_1= np.sum(all_fission_1_MM, all_fission_1)
corr_fusion_1= np.sum(all_fusion_1_MM, all_fusion_1)
corr_fission_2= np.sum(all_fission_2_MM, all_fission_2)
corr_fusion_2= np.sum(all_fusion_2_MM, all_fusion_2)
corr_fission_control= np.sum(all_fission_control_MM, all_fission_control)
corr_fusion_control= np.sum(all_fusion_control_MM, all_fusion_control)

print(f"fission: corr 05 {corr_fission_05}, corr_0{corr_fission_0}, corr1{corr_fission_1}, corr2{corr_fission_2},corr_control{corr_fission_control}  ")
print(f"fusion: corr 05 {corr_fusion_05}, corr_0{corr_fusion_0}, corr1{corr_fusion_1}, corr2{corr_fusion_2},corr_control{corr_fusion_control}  ")
'''
#correlation between 


with open('/home/jirapong/Mito_data/20240810_test.ome.tif_fusion.txt', 'r') as file:
    lines = file.readlines()
data = []
for line in lines:
    line = line.strip()
    row = [0 if x.lower() == 'nan' else int(x) for x in line.split(',')]
    data.append(row)
array = np.array(data)
fusion_number = np.count_nonzero(array,axis = 0)

nellie_df = pd.read_csv(file_path)
labeled_im = tifffile.imread(seg_path)
reassigned_im = tifffile.imread(reassigned_path)

'''
fission_path_05_self = "/home/jirapong/nellie/0.5_glu_output_self_fission.csv"
fusion_path_05_self = "/home/jirapong/nellie/0.5_glu_output_self_fusion.csv"
fission_path_0_self = "/home/jirapong/nellie/0_glu_output_self_fission.csv"
fusion_path_0_self = "/home/jirapong/nellie/0_glu_output_self_fusion.csv"
fission_path_1_self = "/home/jirapong/nellie/1_glu_output_self_fission.csv"
fusion_path_1_self = "/home/jirapong/nellie/1_glu_output_self_fusion.csv"
fission_path_2_self = "/home/jirapong/nellie/2_glu_output_self_fission.csv"
fusion_path_2_self = "/home/jirapong/nellie/2_glu_output_self_fusion.csv"
fission_path_control_self = "/home/jirapong/nellie/control_output_self_fission.csv"
fusion_path_control_self = "/home/jirapong/nellie/control_output_self_fusion.csv"

fission_df_05_self = pd.read_csv(fission_path_05_self)
fusion_df_05_self = pd.read_csv(fusion_path_05_self)
fission_df_0_self = pd.read_csv(fission_path_0_self)
fusion_df_0_self = pd.read_csv(fusion_path_0_self)
fission_df_1_self = pd.read_csv(fission_path_1_self)
fusion_df_1_self = pd.read_csv(fusion_path_1_self)
fission_df_2_self = pd.read_csv(fission_path_2_self)
fusion_df_2_self = pd.read_csv(fusion_path_2_self)
fission_df_control_self = pd.read_csv(fission_path_control_self)
fusion_df_control_self = pd.read_csv(fusion_path_control_self)

fission_event_05_self = np.count_nonzero(fission_df_05_self,axis = 0)
fusion_event_05_self = np.count_nonzero(fusion_df_05_self,axis = 0)
fission_event_0_self = np.count_nonzero(fission_df_0_self,axis = 0)
fusion_event_0_self = np.count_nonzero(fusion_df_0_self,axis = 0)
fission_event_1_self = np.count_nonzero(fission_df_1_self,axis = 0)
fusion_event_1_self = np.count_nonzero(fusion_df_1_self,axis = 0)
fission_event_2_self = np.count_nonzero(fission_df_2_self,axis = 0)
fusion_event_2_self = np.count_nonzero(fusion_df_2_self,axis = 0)
fission_event_control_self = np.count_nonzero(fission_df_control_self,axis = 0)
fusion_event_control_self = np.count_nonzero(fusion_df_control_self,axis = 0)

all_fission_05 = np.sum(fission_event_05_self)
all_fusion_05 = np.sum(fusion_event_05_self)
all_fission_0 = np.sum(fission_event_0_self)
all_fusion_0= np.sum(fusion_event_0_self)
all_fission_1= np.sum(fission_event_1_self)
all_fusion_1= np.sum(fusion_event_1_self)
all_fission_2= np.sum(fission_event_2_self)
all_fusion_2= np.sum(fusion_event_2_self)
all_fission_control= np.sum(fission_event_control_self)
all_fusion_control= np.sum(fusion_event_control_self)
'''