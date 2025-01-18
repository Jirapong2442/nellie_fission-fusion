import scipy.ndimage
from analysis import get_nellie_inputs, get_nellie_outputs
import tifffile
import numpy as np
import pandas as pd
import scipy.stats
import os

from plot_area_label import plot_multiple_line
import matplotlib.pyplot as plt

# what to do 
# 1. self fission fusion for toxivity 
# 2. fission fusion for glu data for fission-fusion ratio

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
    text_box = f'Correlation: {corr_score[0]:.2f}/np-value: {corr_score[1]:.4f}'
    plt.text(0.05, 0.95, text_box, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Show the plot
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

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

def get_fiss_fus_all (path):
    output_fission = []
    output_fusion = []
    fission_df = []
    fusion_df = []

    for file in os.listdir(path):
        if "fission" in file and os.path.isfile(os.path.join(path,file)):
            output_fission.append(os.path.join(path,file))
        if "fusion" in file and os.path.isfile(os.path.join(path,file)):
            output_fusion.append(os.path.join(path,file))

    fission_df,fission_frame = get_fiss_fus(output_fission,fission_df)
    fusion_df,fusion_frame = get_fiss_fus(output_fusion,fusion_df)

    return fission_df, fusion_df, fission_frame , fusion_frame
    
def calculate_fiss_fus_ratio(fission_df, fusion_df):
    fiss_fus_ratios = []
    for i in range(len(fusion_df)):
        for j in range(len(fusion_df[i])):
            try:
                fiss_fus_ratios.append(fission_df[i][j] / fusion_df[i][j])
            except ZeroDivisionError:
                fiss_fus_ratios.append(0)
    return fiss_fus_ratios

def plot_two_axis(fpr_list, tpr_list, plot_label_list, names, title="Multiple ROC Curves" ):
    """
    Plot multiple ROC curves on the same plot with different colors
    
    Parameters:
    fpr_list: list of lists/arrays containing FPR values for each curve
    tpr_list: list of lists/arrays containing TPR values for each curve
    plot_label_list: list of lists containing point labels for each curve
    names: list of names for each ROC curve (will appear in legend)
    title: title of the plot
    """
    # Create color map for different curves
    colors = plt.cm.rainbow(np.linspace(0, 1, len(fpr_list)))
    
    # Create the plot
    #plt.figure(figsize=(12, 8))
    fig, ax1 = plt.subplots()
    
    # Plot each ROC curve
    for i, (fpr, tpr, labels, name, color) in enumerate(zip(fpr_list[0:-1], tpr_list[0:-1], plot_label_list[0:-1], names[0:-1], colors)):
        # Plot connected lines
        ax1.plot(fpr, tpr, color=color, alpha=0.5, label=f'{name} (Path)')
        
        # Plot points
        ax1.scatter(fpr, tpr, color=color, s=100, alpha=0.6, label=f'{name} (Points)')
        
        # Add labels for points
        """
        for j, label in enumerate(labels):
            plt.annotate(f'{label}',
                        (fpr[j], tpr[j]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        color=color,
                        alpha=0.7)
        """
    
    # Add diagonal line
    #plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    
    # Add grid, labels, and title
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Number of frame')
    ax1.set_ylabel('number of label')
    
    ax2 = ax1.twinx()
    ax2.plot(fpr_list[-1], tpr_list[-1], color="red", label=f'{names[-1]} (Path)')
    ax2.set_ylabel("number of component", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    
    plt.title(title)
   
    
    # Adjust legend to remove duplicate point entries
    handles, labels = ax1.get_legend_handles_labels()#plt.gca()
    handles2, labels2 = ax2.get_legend_handles_labels()#plt.gca()
    unique_labels = []
    unique_handles = []
    seen_names = set()
    
    handles = handles + handles2
    labels = labels + labels2
    for handle, label in zip(handles, labels):
        name = label.split(' (')[0]
        if name not in seen_names:
            seen_names.add(name)
            unique_labels.append(name)
            unique_handles.append(handle)
    
    plt.legend(unique_handles, unique_labels, loc='upper right')
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    dir_path_tox = "./nellie_output/toxicity"
    dir_path_mdivi = "./nellie_output/mdivi"

    dir_path_mdivi_self = "./self_event/mdivi/"
    dir_path_tox_self = "./self_event/toxicity/"

    mitometer_path = 'D:/Internship/NTU/algo_output/mito'
    mito_fission = []
    mito_fusion = []

    mito_fiss_all = []
    mito_fus_all = []

    column_tox = os.listdir(dir_path_tox)
    column_glu = os.listdir(dir_path_mdivi)
    '''
    for file in os.listdir(mitometer_path):
    # check if current path is a file
        if "fission" in file and os.path.isfile(os.path.join(mitometer_path,file)):
            mito_fission.append(os.path.join(mitometer_path,file))
        if "fusion" in file and os.path.isfile(os.path.join(mitometer_path,file)):
            mito_fusion.append(os.path.join(mitometer_path,file))

    for file in mito_fission:
        fission_all = check_fission_fusion_MM(file)
        if len(mito_fiss_all) == 0:
            mito_fiss_all = [fission_all]
        else:
            mito_fiss_all.append(fission_all)


    for file in mito_fusion:
        fission_all = check_fission_fusion_MM(file)
    '''

    
    fission_tox, fusion_tox, fiss_frame, fus_frame = get_fiss_fus_all(dir_path_tox)
    fiss_fus_ratios_tox = calculate_fiss_fus_ratio(fission_tox, fusion_tox)
    

    fission_tox_self, fusion_tox_self, fiss_self_frame, fus_self_frame = get_fiss_fus_all(dir_path_tox_self)
    fiss_fus_ratios_tox_self = calculate_fiss_fus_ratio(fission_tox_self, fusion_tox_self)

    #order: control 10min, control 3, mdivi 10min, mdivi 3
    fission_mdivi, fusion_mdivi, fiss_frame_mdivi, fus_frame_mdivi = get_fiss_fus_all(dir_path_mdivi)
    fiss_fus_ratios_mdivi = calculate_fiss_fus_ratio(fission_mdivi, fusion_mdivi)

    fission_mdivi_self, fusion_mdivi_self, fiss_self_frame_mdivi, fus_self_frame_mdivi = get_fiss_fus_all(dir_path_mdivi_self)
    fiss_fus_ratios_mdivi_self = calculate_fiss_fus_ratio(fission_mdivi_self, fusion_mdivi_self)

    #combine self fission-fusion with normal one
    fiss_tox_all = [list(fission_tox[i][j] + fission_tox_self[i][j] for j in range(len(fission_tox_self[i]))) for i in range(len(fission_tox))]
    fusion_tox_all = [list(fusion_tox[i][j] + fusion_tox_self[i][j] for j in range(len(fusion_tox_self[i]))) for i in range(len(fusion_tox))]
    fiss_fus_ratios_tox_all = calculate_fiss_fus_ratio(fiss_tox_all, fusion_tox_all)
    
    #all connected components

    #check dims and verify that it's correct
    all_label = []
    all_area = []
    all_index = []
    all_raw_label = []
    label_path = "./check_label&area/mdivi/"

    files= os.listdir(label_path)
    all_name = [files[i][0:5] for i in range(len(files))]

    for file in files:
        df = pd.read_csv(label_path + file)
        index = np.expand_dims(np.arange(len(df)),axis=1)
        
        df_label = np.array(df[['label_num']])
        #df_label = np.hstack((index, df_label))

        df_area = np.array(df[['area']])
        #df_area = np.hstack((index, df_area))

        df_raw_label = np.array(df[['raw_label']])

        all_label.append(df_label)
        all_area.append(df_area)
        all_index.append(index)
        all_raw_label.append(df_raw_label)
        
        '''
        try: 
            all_label = np.vstack((all_label, df_label))
            all_area = np.vstack((all_area, df_area))

        except ValueError:
            all_label = df_label
            all_area = df_area
        '''


    index_tox = [len(fusion_tox[x]) for x in range(len(fusion_tox)) ]
    #index_glu = [len(fusion_glu[x]) for x in range(len(fusion_glu)) ]

    #fiss fusion ratio of toxicity
    all_stat_control = fiss_fus_ratios_tox[1:index_tox[0]]
    all_stat_FCCP = fiss_fus_ratios_tox[index_tox[0]+1: np.sum(index_tox[0:2])]
    all_stat_oligo = fiss_fus_ratios_tox[np.sum(index_tox[0:2])+1  : np.sum(index_tox[0:3])]
    all_stat_Rotenone = fiss_fus_ratios_tox[np.sum(index_tox[0:3])+1  :np.sum(index_tox[0:4])]

    all_stat_control_log = np.log(all_stat_control)
    all_stat_FCCP_log = np.log(all_stat_FCCP)
    all_stat_oligo_log = np.log(all_stat_oligo)
    all_stat_Rotenone_log = np.log(all_stat_Rotenone)

    #self fiss fusion ratio of toxicity
    all_stat_control_self = fiss_fus_ratios_tox_self[1:index_tox[0]]
    all_stat_FCCP_self = fiss_fus_ratios_tox_self[index_tox[0]+1: np.sum(index_tox[0:2])]
    all_stat_oligo_self = fiss_fus_ratios_tox_self[np.sum(index_tox[0:2])+1  : np.sum(index_tox[0:3])]
    all_stat_Rotenone_self = fiss_fus_ratios_tox_self[np.sum(index_tox[0:3])+1  :np.sum(index_tox[0:4])]

    all_stat_control_self_log = np.log(all_stat_control_self)
    all_stat_FCCP_self_log = np.log(all_stat_FCCP_self)
    all_stat_oligo_self_log = np.log(all_stat_oligo_self)
    all_stat_Rotenone_self_log = np.log(all_stat_Rotenone_self)

    #all fiss fusion ratio of toxicity
    all_stat_control_all = fiss_fus_ratios_tox_all[1:index_tox[0]]
    all_stat_FCCP_all = fiss_fus_ratios_tox_all[index_tox[0]+1: np.sum(index_tox[0:2])]
    all_stat_oligo_all = fiss_fus_ratios_tox_all[np.sum(index_tox[0:2])+1  : np.sum(index_tox[0:3])]
    all_stat_Rotenone_all = fiss_fus_ratios_tox_all[np.sum(index_tox[0:3])+1  :np.sum(index_tox[0:4])]

    all_stat_control_all_log = np.log(all_stat_control_all)
    all_stat_FCCP_all_log = np.log(all_stat_FCCP_all)
    all_stat_oligo_all_log = np.log(all_stat_oligo_all)
    all_stat_Rotenone_all_log = np.log(all_stat_Rotenone_all)


    #plot fission fusion by frame
    #classify by toxin type
    # shift data of 0 to the right
    ending = 140
    starting = 2
    control =   [fission_tox[0][1:ending]] + [fusion_tox[0][1:ending]] + [all_stat_control_all[starting-1:ending-1]] #+ [all_raw_label[0][starting:ending]] #
    control_frame  =  [fiss_frame[0][1:ending]] + [fus_frame[0][1:ending]] +  [fiss_frame[0][starting:ending]] #+ [fiss_frame[0][starting:ending]]#

    FCCP =  [fission_tox[1][1:ending]] + [fusion_tox[1][1:ending]] +[all_stat_FCCP_all[starting-1:ending-1]] #+ [all_raw_label[1][starting:ending]] #
    FCCP_frame = [fiss_frame[1][1:ending]] + [fus_frame[1][1:ending]] +   [fiss_frame[1][starting:ending]] #+ [fiss_frame[1][starting:ending]]#

    oligo =    [fission_tox[2][1:ending]] + [fusion_tox[2][1:ending]] + [all_stat_oligo_all[starting-1:ending-1]] #+ [all_raw_label[2][starting:ending]]#
    oligo_frame = [fiss_frame[2][1:ending]] + [fus_frame[2][1:ending]] + [fiss_frame[2][starting:ending]] #+  [fiss_frame[2][starting:ending]]#

    Rotenone = [fission_tox[3][1:ending]] + [fusion_tox[3][1:ending]] + [all_stat_Rotenone_all[starting-1:ending-1]] #+ [all_raw_label[3][starting:ending]] #
    Rotenone_frame =  [fiss_frame[3][1:ending]] + [fus_frame[3][1:ending]] + [fiss_frame[3][starting:ending]]#+ [fiss_frame[3][starting:ending]]#
    
    #plot_two_axis(Rotenone_frame,Rotenone, Rotenone_frame, [  'fission-fusion ratio', 'all component'] , "number of component in Rotenone")
    plot_multiple_line(Rotenone_frame,Rotenone, Rotenone_frame, [ 'fission' , 'fusion', 'fission-fusion ratio'] , "number of component in Rotenone")

    plt.boxplot([all_stat_control_all_log,all_stat_FCCP_all_log, all_stat_oligo_all_log,all_stat_Rotenone_all_log], labels=[ 'control','FCCP', 'oligo', 'Rotenone'])

    # Adding a title and labels
    plt.title('fission fusion ratio of different toxins')
    plt.ylabel('Values')

    # Display the plot
    plt.show()

    '''
       # list to store files
    output_fission = []
    output_fusion = []
    fission_df = []
    fusion_df = []

    # Iterate directory
    # order: control FCCP oligo rotenone
    for file in os.listdir(dir_path):
    # check if current path is a file
        if "output_fission" in file and os.path.isfile(os.path.join(dir_path,file)):
            output_fission.append(os.path.join(dir_path,file))
        if "output_fusion" in file and os.path.isfile(os.path.join(dir_path,file)):
            output_fusion.append(os.path.join(dir_path,file))

    fission_df = get_fiss_fus(output_fission,fission_df)
    fusion_df = get_fiss_fus(output_fusion,fusion_df)


    fiss_fus_ratios = []

    for i in range(len(fusion_df)):
        for j in range(len(fusion_df[i])):
            try:
                fiss_fus_ratios.append(fission_df[i][j] / fusion_df[i][j])
            except ZeroDivisionError:
                fiss_fus_ratios.append(0)
    '''