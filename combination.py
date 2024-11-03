# find comb
import numpy as np
import pandas as pd

area_path = "D:/Internship/NTU/algo_output/Start_1st_frame/1_1_neighbour.csv"
df = pd.read_csv(area_path)
mini_df = df[["Volume_pre", "Volume_post", "Nearest Label","Distance","Frame","Label"]]

#run only for 1 event of two datapoint 
# segment input into a correspoding label and put it into the function
def find_combinations(arr,error_percentage=0.15,start=1,):
    '''
    find a combination of first row that could potentially sum up to the target value in second row
    return value pair data of
        1. a possible pair data of data_column to the specific_cell_value (result data)
        2. a possible pari data of target_column to the specific_cell_value (result_target)
    
        Equation:
        specific_cell_value + pre_event_arr = target_col + result_target
    '''

    # mini_df = df[["Volume_pre", "Volume_post", "Nearest Label","Distance","Frame","Label"]]
    pre_event_arr = []
    post_event_arr = []
    result_label = []
    result_dists = []
    combination_set= []
    error_range = []
    dists_col = arr[:,3]
    label_col = arr[:,2]

    if arr[0,1] < arr[0,0]:
        data_col = arr[:, 1]
        target_col = arr[:, 0]  

    else :
        data_col = arr[:, 0]
        target_col = arr[:, 1]


    target_value = target_col[0]
    specific_cell_value = data_col[0]
    remaining_target = target_value - specific_cell_value
    sum_all_vol = target_value+specific_cell_value
    combination_num = 0

    upper_boundary, lower_boundary = error_percentage*target_value, -error_percentage*target_value

    def backtrack(data_list,target_list,label_list,dist_list, start, remaining,sum, combination_num):
       #if (remaining/ (sum/2)) <= error_percentage:
        if (remaining/ (sum/2)) <= 0.5 and len(data_list) > 0  : #check all combination
        #remaining >= lower_bound and remaining <= upper_bound :
            #remaining inbetween this range == acceptable
            pre_event_arr.append(data_list[:])
            post_event_arr.append(target_list[:])
            result_label.append(label_list[:])
            result_dists.append(dist_list[:])

            combination_arr = np.full(shape=(len(dist_list)) , fill_value = combination_num)
            combination_set.append(combination_arr)

            error_num = remaining/ (sum/2)
            error = np.full(shape=(len(dist_list)) , fill_value = error_num)
            error_range.append(error)
            return
        
        for i in range(start, len(data_col)):
            #second condition after and allow it to select area that only decrease
            if target_col[i] < data_col[i]:#remaining - data_col[i] >= lower_boundary and 
                data_list.append(data_col[i])
                target_list.append(target_col[i])
                label_list.append(label_col[i])
                dist_list.append(dists_col[i])

                backtrack(data_list,target_list,label_list,dist_list, i + 1, remaining - (data_col[i]-target_col[i]),
                           sum + data_col[i] + target_col[i],combination_num)
                combination_num = combination_num + 1
                data_list.pop()
                target_list.pop()
                label_list.pop()
                dist_list.pop()
    #when the lable suddenly appear or disappear = specific_cell_volume = 0 -> break the combination
   
    if specific_cell_value == 0: 
        return np.array([],[],[])

    else:
        backtrack([],[],[], [],start, remaining_target,sum_all_vol, combination_num)
        volume_pre = np.array(pre_event_arr,dtype=object)
        volume_post = np.array(post_event_arr,dtype=object)
        label = np.array(result_label,dtype=object)
        dist = np.array(result_dists,dtype=object)
        set_all = np.array(combination_set,dtype=object)
        error_all = np.array(error_range,dtype=object)

    #dim are the same
    #result = np.array([volume_pre,volume_post,dist,label], dtype = object)
    result = np.transpose(np.concatenate((volume_pre,volume_post,dist,label)))

    return result,set_all,error_all


check_frame = 0
check_label = 304
working_arr = mini_df[mini_df["Frame"] == check_frame]
neighbor = np.array(working_arr[working_arr["Label"] == check_label])
combinations,combination_set,error_all = find_combinations(neighbor)
try:
    
    if combinations.shape[1] != 0:
        section_index = int((combinations.shape[1])/4)

        pre_event_volume = combinations[:,:section_index]
        post_event_volume = combinations[:,section_index:section_index*2]
        dists = combinations[:,2*section_index:3*section_index]
        label_arr= combinations[:,3*section_index:4*section_index]
        pre_event_volume = np.array(pre_event_volume,dtype=np.float64)
        post_event_volume = np.array(post_event_volume,dtype=np.float64)
        dists = np.array(dists,dtype=np.float64)
        label_arr = np.array(label_arr,dtype=np.float64)
    else:
        print("no")
        
except:
    combinations = np.concatenate(combinations)
    section_index = int(len(combinations)/4)
    pre_event_volume =  np.concatenate(np.expand_dims(combinations[:section_index],axis=0))
    post_event_volume = np.concatenate(np.expand_dims(combinations[section_index:2*section_index],axis=0))
    dists = np.concatenate(np.expand_dims(combinations[2*section_index:3*section_index],axis=0))
    label_arr = np.concatenate(np.expand_dims(combinations[3*section_index:4*section_index],axis=0))

arr = np.concatenate((np.expand_dims(pre_event_volume,axis = 1),
                      np.expand_dims(post_event_volume,axis = 1),
                      np.expand_dims(dists,axis = 1),
                      np.expand_dims(label_arr,axis = 1)), 
                      #np.expand_dims(np.concatenate(combination_set),axis = 1),
                      #np.expand_dims(np.concatenate(error_all),axis = 1)),
                        axis = 1,dtype=np.float64)
print(arr)
'''
TO RUN ALL FRAME

for frame in pd.unique(mini_df["Frame"]):
    working_arr = mini_df[mini_df["Frame"] == frame]

    for label in pd.unique(working_arr["Label"]):

        neighbor = np.array(working_arr[working_arr["Label"] == label])
        all_combination = find_combinations(neighbor)
        print()
'''
