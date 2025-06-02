import numpy as np
import pandas as pd
import os
from checkFissFus2 import get_fiss_fus_all  , fiss_fus_reassigned_label
# get_fission_fus_all => output all fission fusion of that folder (control,FCCP,oligomycin, rotenone) row 
# column = frame
from plot_area_label import plot_multiple_line
import matplotlib.pyplot as plt
import pingouin as pg
import pandas as pd

"""
hyperparameters: neighbor = 3, area difference 0.25, combination difference = 23
self-fission/fusion = incorporate all label not occurance
"""
def ANOVA_test (measured_stat):
    k = len(measured_stat)
    N = len(measured_stat[0])

    group_means = np.mean(measured_stat, axis=1) # control, FCCP, oligo, rotenone
    total_group_mean = np.mean(group_means)
    timepoint_means = np.mean(measured_stat, axis=0)

    SSB = N * np.sum((group_means - total_group_mean)**2)
    SSW = np.sum((measured_stat - group_means[:, np.newaxis])**2)
    SSS = k * np.sum((total_group_mean - timepoint_means)**2, axis=0)
    #maybe each time point carries no information
    SSE = SSW - SSS

    MS_group = SSB / (k - 1)
    MS_error = SSE / ((k-1) * (N - 1))
    F = MS_group / MS_error

    return np.append(group_means,F)

def check_var (stat):
    var1 = np.var(stat[0])
    var2 = np.var(stat[1])
    F = var1/var2
    return F

def t_test (stat):
    mean1 = np.mean(stat[0])
    mean2 = np.mean(stat[1])
    var1 = np.var(stat[0])
    var2 = np.var(stat[1])
    n1 = len(stat[0])
    n2 = len(stat[1])

    var_all = ((n1-1) * var1 + (n2-1) * var2)/(n1+n2-2)
    t = (mean1 - mean2) / np.sqrt(var_all * (1/n1 + 1/n2))

    return np.append([mean1,mean2],t)

def sliding_window_cumsum(arr, window_size):
    result = []
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        result.append(np.sum(arr[start:end]))
    return np.array(result)

def regularization(arr, k_val):
    try: 
        min_val = np.min(arr[ arr> 0])
    except ValueError: # when min == 0
        min_val = 1
    epsilon = min_val * k_val
    arr = np.where(arr == 0, epsilon,arr)
    return arr

#return as a tuple to unpack later

def apply_function_to_arrays(arrays, func, *args):
    '''
def apply_function_to_arrays(arrays, func, *args):
    return np.array([func(arr, *args) for arr in arrays])'''
    # Apply func to each array; capture results as a list of tuples (if multiple outputs)
    results = [func(arr, *args) for arr in arrays]
    
    # Check if the first result is a tuple (i.e., func returns multiple outputs)
    if isinstance(results[0], tuple):
        # Transpose the results to group each output type
        transposed_results = list(zip(*results))
        # Convert each group to a numpy array
        return tuple(np.array(group) for group in transposed_results)
    else:
        # Single output: return a single numpy array
        return np.array(results)

def apply_function_to_path(func, *args):
    results = func(*args)
    return results

def apply_function_folder(*args, keywords, func,) :
    all_results = {}
    for file in os.listdir(*args):
        if keywords in file and os.path.isfile(os.path.join(*args,file)):
            results = apply_function_to_path(func, os.path.join(*args,file))
            # Check if the first result is a tuple (i.e., func returns multiple outputs)
            all_results[file] = results
    return all_results

def convert_to_sig_test(df, col_value, regularization_ = False, epsilon_ = 0.001):

    if regularization_:
        df = apply_function_to_arrays(df, regularization, epsilon_).flatten()
    else:
        df = df.flatten()
    pan_df = pd.DataFrame({ "toxin": col_value,'event': df}) 
    return df,pan_df
        
        

if __name__ == "__main__":
    dir_path_tox = "./nellie_output/toxicity/0.25"
    dir_path_mdivi = "./nellie_output/mdivi/0.25"

    dir_path_mdivi_self = "./self_event/mdivi/num"
    dir_path_tox_self = "./self_event/toxicity/num"

    dir_tox_meter = "D:/Internship/NTU/my_script/mitometer_output/toxin"
    dir_mdivi_meter = "D:/Internship/NTU/my_script/mitometer_output/mdivi"

    dir_raw_tox = "D:/Internship/NTU/nellie_output/nellie_output/toxins/time_ins_FCCP.ome-ch0-features_components.csv"
    dir_raw_fold = "D:/Internship/NTU/nellie_output/nellie_output/toxins/"
    
    #output = apply_function_to_path(fiss_fus_reassigned_label, dir_raw_tox)
    #fold_out = apply_function_folder(dir_raw_fold, "features_components",fiss_fus_reassigned_label)
    

    #mitometer
    fission_tox_meter, fusion_tox_meter, fiss_frame_meter, fus_frame_meter = get_fiss_fus_all(dir_tox_meter)
    fission_mdivi_meter, fusion_mdivi_meter, fiss_frame_meter_tox, fus_frame_meter_tox = get_fiss_fus_all(dir_mdivi_meter)
    fusion_mdivi_meter[2] = fusion_mdivi_meter[2][0:61]
    fission_mdivi_meter[2] = fission_mdivi_meter[2][0:61]

    # get fission fusion result from nellie output
    fission_tox, fusion_tox, fiss_frame, fus_frame = get_fiss_fus_all(dir_path_tox)
    fission_tox_self, fusion_tox_self, fiss_self_frame, fus_self_frame = get_fiss_fus_all(dir_path_tox_self , isProb= True)

    # control 10min, control 3, mdivi 10min, mdivi 3
    fission_mdivi, fusion_mdivi, fiss_frame_mdivi, fus_frame_mdivi = get_fiss_fus_all(dir_path_mdivi ) 
    fission_mdivi_self, fusion_mdivi_self, fiss_self_frame_mdivi, fus_self_frame_mdivi = get_fiss_fus_all(dir_path_mdivi_self , isProb= False)

    # 10 min fission
    fission_mdivi_10minimum = np.array([fission_mdivi_meter[0],fission_mdivi_meter[2] ])
    # 1.3 second fission
    fission_mdivi_meter = [fission_mdivi_meter[1],fission_mdivi_meter[3] ]
    minimum = np.min([len(x) for x in fission_mdivi_meter])
    fission_mdivi_meter = np.array([x[0:minimum] for x in fission_mdivi_meter])
    MA_fusion_mdivi_mito_10min = apply_function_to_arrays(fission_mdivi_10minimum, regularization, 0.01)

    # 10 min fusion
    fusion_mdivi_10min = np.array([fusion_mdivi_meter[0],fusion_mdivi_meter[2] ])
    # 1.3 second fusion
    fusion_mdivi_meter = [fusion_mdivi_meter[1],fusion_mdivi_meter[3] ]
    minimum = np.min([len(x) for x in fusion_mdivi_meter])
    fusion_mdivi_meter = np.array([x[0:minimum] for x in fusion_mdivi_meter])

    # 10 min self_fission
    fission_mdivi_self_10min = np.array([fission_mdivi_self[0],fission_mdivi_self[2] ])
    # 1.3 second self_fission
    fission_mdivi_self = [fission_mdivi_self[1],fission_mdivi_self[3] ]
    minimum = np.min([len(x) for x in fission_mdivi_self])
    fission_mdivi_self = np.array([x[0:minimum] for x in fission_mdivi_self])

    # 10 min self_fusion
    fusion_mdivi_self_10min = np.array([fusion_mdivi_self[0],fusion_mdivi_self[2] ])
    # 1.3 second self_fusion
    fusion_mdivi_self = [fusion_mdivi_self[1],fusion_mdivi_self[3] ]
    minimum = np.min([len(x) for x in fusion_mdivi_self])
    fusion_mdivi_self = np.array([x[0:minimum] for x in fusion_mdivi_self])


    #fission_mdivi_all = fission_mdivi + fission_mdivi_self
    #fusion_mdivi_all = fusion_mdivi + fusion_mdivi_self

    #fission_mdivi_10min_all = fission_mdivi_10min + fission_mdivi_self_10min
    #fusion_mdivi_10min_all = fusion_mdivi_10min + fusion_mdivi_self_10min

    # toxins
    minimum = np.min([len(x) for x in fission_tox])
    fission_tox = np.array([x[0:minimum] for x in fission_tox])
    fusion_tox = np.array([x[0:minimum] for x in fusion_tox])

    fission_tox_self = np.array([x[0:minimum] for x in fission_tox_self])
    fusion_tox_self = np.array([x[0:minimum] for x in fusion_tox_self])

    fission_tox_meter = np.array([x[0:minimum] for x in fission_tox_meter])
    fusion_tox_meter = np.array([x[0:minimum] for x in fusion_tox_meter])


    # data preparation with regularization (0 -> epsilon)
    # window == 62
    MA_fusion_tox = apply_function_to_arrays(fusion_tox, regularization, 0.01)
    MA_fission_tox = apply_function_to_arrays(fission_tox, regularization, 0.01)

    MA_fusion_tox_self = apply_function_to_arrays(fusion_tox_self, regularization, 0.01)
    MA_fission_tox_self = apply_function_to_arrays(fission_tox_self, regularization, 0.01)

    MA_fusion_tox_mito = apply_function_to_arrays(fusion_tox_meter, regularization, 0.01)
    MA_fission_tox_mito = apply_function_to_arrays(fission_tox_meter, regularization, 0.01)

    fission_all = MA_fission_tox + MA_fission_tox_self
    fusion_all = MA_fusion_tox + MA_fusion_tox_self   


    all_label = []
    all_area = []
    all_index = []
    all_raw_label = []
    label_path = "./check_label&area/toxicity/"

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
    #for mdivi
    area_10min = np.squeeze(np.array([all_area[0],all_area[2] ]))
    area_1s = [all_area[1],all_area[3] ]
    minimum = np.min([len(x) for x in area_1s])
    area_1s = np.squeeze(np.array([x[0:minimum] for x in area_1s]))
    raw_label_10min = np.squeeze(np.array([all_raw_label[0],all_raw_label[2] ]))
    raw_label_1s = [all_raw_label[1],all_raw_label[3] ]
    minimum = np.min([len(x) for x in raw_label_1s])
    raw_label_1s = np.squeeze(np.array([x[0:minimum] for x in raw_label_1s]))

    #fiss_per_area = fission_mdivi_all / area_1s 
    #fiss_per_area_10min = fission_mdivi_10min_all / area_10min

    #fiss_per_component = fission_mdivi_all / raw_label_1s
    #fiss_per_component_10min = fission_mdivi_10min_all / raw_label_10min

    fusion_mdivi_10min = np.where(fusion_mdivi_10min == 0, 1, fusion_mdivi_10min)
    fusion_mdivi_meter = np.where(fusion_mdivi_meter == 0, 1, fusion_mdivi_meter)
    measured_stat = fusion_mdivi_meter
    significance_mito_fission = t_test(fission_mdivi_10min/fusion_mdivi_10min)
    significance_mito_fusion = t_test(fission_mdivi_meter/fusion_mdivi_meter)
    measured_stat2 = fusion_mdivi_10min
    '''
    # for toxins
    all_area = np.squeeze(np.array([x[0:minimum] for x in all_area]))
    all_raw_label = np.squeeze(np.array([x[0:minimum] for x in all_raw_label]))
    
    measured_stat = MA_fission_tox
    x =np.expand_dims(np.arange(len(MA_fission_tox[0])), axis = 0)
    frame=  np.repeat(x,4,axis = 0)
    #for fiss fus ratio
    #measured_stat = (fission_tox+1) / (fusion_tox+1) #+1 to avoid division by 0
    #measured_stat[measured_stat]
    #measured_stat = np.nan_to_num(measured_stat)

    log_measure_stat = np.log(measured_stat)
    log_measure_stat[np.isinf(log_measure_stat)] = 0

    

    plot_multiple_line(frame,MA_fission_tox, frame, ['control', 'FCCP', 'oligomycin', 'Rotenone'] , "number of fission/fusion MDIVI 10 min")

    #plt.boxplot([measured_stat[0],measured_stat[1] , measured_stat[2], measured_stat[3]], labels=[ 'control', 'FCCP', 'oligomycin', 'Rotenone'])
    #plt.boxplot([measured_stat[0],measured_stat[1] , measured_stat2[0], measured_stat2[1]], labels=[ 'control', 'mdivi', 'control 10min', 'mdivi 10min'])

    # Adding a title and labels
    plt.title('Total Fission value')
    #plt.title('Fusion per area')
    #plt.title('Fusion per component') 

    #plt.title('Fission-fusion ratio') 
    #plt.ylabel('Number of Fission events')
    plt.ylabel('Number of Fusion events')



    # Display the plot
    plt.show()

 
