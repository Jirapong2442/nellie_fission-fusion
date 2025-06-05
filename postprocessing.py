import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import center_of_mass   
from scipy.spatial import cKDTree
import tifffile
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
num_neighbour = 3

def check_Fis_Fus_Mitometer (file_path,timeDim):
    #check fission fusion from mitometer output
    file_path = file_path
    c = np.zeros((timeDim,))
    event_per_frame = np.zeros((timeDim,))
    frame_length = timeDim
    i = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.replace('NaN', '0')
        arr = np.fromstring(line, dtype=int, sep=', ')

        if i == 0:
            c = np.stack((arr, c),axis=0)
        else: 
            c = np.vstack((c,arr))
        i += 1 
    #column =frame row =label
    # value = track number from which mitochondria fission 
    np_array = np.array(c)
    for frame in range(frame_length):
        event_per_frame[frame] = np.count_nonzero(np_array[0:,frame])

    return event_per_frame

def check_mito_number(df, label, final_fission_self, final_fusion_self):
    max_frame_num = int(df['t'].max()) + 1

    for i in range(max_frame_num):
        area_in_frame = df[df['t'] == i]["organelle_area_raw"]
        std = np.std(area_in_frame)
        accepted_area = np.mean(area_in_frame)

        new_label = df[df['reassigned_label_raw'] == label ]
        new_label = new_label[new_label['t'] == i]
        test_area = np.sum(new_label["organelle_area_raw"])
        current_frame_fragment = new_label.shape[0]
        
        #add area threshold
        if test_area >= accepted_area:
            if i == 0:
                test_fragment = current_frame_fragment

            elif current_frame_fragment > test_fragment and current_frame_fragment > 0 and test_fragment > 0:
                #final_fission_self[label-1][i] = final_fission_self[label-1][i] + 1
                final_fission_self[label-1][i] = current_frame_fragment - test_fragment
                test_fragment = current_frame_fragment 

            elif current_frame_fragment < test_fragment and current_frame_fragment > 0 and test_fragment > 0:
                #final_fusion_self[label-1][i] = final_fusion_self[label-1][i] + 1
                final_fusion_self[label-1][i] = test_fragment - current_frame_fragment
                test_fragment = current_frame_fragment
            
            elif (current_frame_fragment > test_fragment and test_fragment == 0) or (current_frame_fragment < test_fragment and current_frame_fragment == 0) or (current_frame_fragment == test_fragment):
                continue
        else:
            test_fragment = current_frame_fragment

    return final_fission_self,final_fusion_self

def find_extrema(binary_image):
    # Ensure the image is binary
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8) * 255
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Get the largest contour
    cnt = max(contours, key=cv2.contourArea)
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    # Calculate extrema points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    # Calculate additional points
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_right = (x + w, y + h)
    bottom_left = (x, y + h)
    # Combine all points
    extrema = np.array([
        top_left,
        topmost,
        top_right,
        rightmost,
        bottom_right,
        bottommost,
        bottom_left,
        leftmost
    ])
    extrema = np.fliplr(extrema) #flip to y,x
    return extrema

def move_row_to_first(arr, value): 
    row_index = np.where((arr[:,3] == value))[0] 

    if len(row_index) == 0: 
        print(f"Value {value} not found in the array.") 
        return arr 

    # Get the first occurrence if multiple matches 
    #row_index = row_index[0] 
    row_of_interest = arr[row_index] 
    remaining_rows = np.delete(arr, row_index, axis=0) 
    
    return np.vstack((row_of_interest, remaining_rows)) 

def nearest_neighbour(labeled_im,csv_output,frame):
    '''
    find closet neighbour from extrema of neigbor label to the centriod of interested label of inoput frame
    input:
        1. labeled_im = labeled image
        2. csv_output = nellie output
        3. frame = frame number

    output:
        1. treeMatrix = [Y,X,label] centriod of 1 label (can be > 1 structure) and label 
        2. all_dists = [dists] distance from the closet centriod of neighbour to interested label extremas
                > (N label of current frame, 8 extermas,, 3 NN)
        3. all_idxs = [idxs] index in tree representing the closet centriod of neighbour from interested label extremas
                > (N label of current frame, 8 extermas,, 3 NN)
        4. pointer to tree object
    '''
    all_dists = []
    all_idxs = []
    centroids = []
    labels = []
    extremas = np.zeros((0,2))
    nellie_df = csv_output
    labeled_im_frame = labeled_im[frame]
    raw_label = np.unique(labeled_im_frame)
    raw_label = raw_label[raw_label != 0]

    # !!!!!cannot find a label since it always start over if we use range(len(unique_label))
    for label in raw_label:
        #find the centriod on all labels after connected component segmentation
        #center of mass function give Y then X of NN element 
        centroid = center_of_mass(labeled_im_frame == label)
        centroids.append(centroid)
        extremas = np.concatenate((extremas,find_extrema(labeled_im_frame == label)),axis = 0)

        #find reassignment label on all label
        nellie_df_2d_label = nellie_df[nellie_df['label'] == label]
        reassigned_label = nellie_df_2d_label['reassigned_label_raw']
        labels.append(int(list(reassigned_label)[0]))

    centroids = np.array(centroids) # centriods of all unique label
    #labels = reassigned label from the unique label. 
    labels = np.expand_dims(np.array(labels), axis=1)
    raw_label = np.expand_dims(raw_label, axis=1)
    final_labels = np.repeat(labels,8,axis=0)
    final_raw_label = np.repeat(raw_label,8,axis=0)
    extrema_treeMatrix = np.hstack((extremas,final_labels,final_raw_label))
    treeMatrix = np.hstack((centroids,labels,raw_label))

    tree = cKDTree(centroids)
    # loop again to get dist and idxs from all label
    # we need complete information of centroid to create a tree => run 2 loop

    for label in raw_label: 
        extrema_loc = extrema_treeMatrix[extrema_treeMatrix[:,3]==label][:,:2]
        #query k closet object
        #find all centriod that closest to the extrema of this label
        dists, idxs = tree.query(np.expand_dims(extrema_loc,axis = 0), k=num_neighbour, workers=-1)
        dists = np.transpose(dists,(1,0,2))
        idxs = np.transpose((idxs),(1,0,2))

        idxs = np.squeeze(idxs)
        dists = np.squeeze(dists)

        all_dists.append(dists.astype(object))
        all_idxs.append(idxs.astype(object))

    all_dists = np.squeeze(np.array(all_dists,dtype=object))
    all_idxs = np.squeeze(np.array(all_idxs,dtype=object))

    #doesnt allow repeatition of self > 1 if happen increase neighbour and pop repeated value??
    return treeMatrix,all_dists,all_idxs

def repeated_value_NN(label,NN,treeMatrix,tree,num_neighbour):

    new_NN = np.delete(NN, np.where(NN==label))
    new_NN = np.insert(new_NN,0,label)
    
    new_neighbour_needed = num_neighbour-new_NN.shape[0]

    centriod_of_label = treeMatrix[treeMatrix[:,2]==label][:,:2]
    avg_centriod = np.average(centriod_of_label,axis = 0)
    #query two closet object
    new_dists, new_idxs = tree.query(np.expand_dims(avg_centriod,axis = 0), k=num_neighbour+new_neighbour_needed, workers=-1)

    return new_dists,new_idxs

def check_volume_value(startingframe,endingframe,label,isFusion,percent_threshold ):

    volume_all = []
    volume_value = []
    raw_label = []
    checkframe = abs(endingframe- startingframe)
    if isFusion:
        for i in range(checkframe+1):
            # change made: area of raw label instead of reassigned label
            new_label = nellie_df[nellie_df['reassigned_label_raw'] == label ]
            new_label = new_label[new_label['t']== i+startingframe]
            volume = np.array(list(new_label['organelle_area_raw']))
            r_label = np.array(list(new_label['label']))
            volume_all.append(volume) 
            raw_label.append(r_label)
    else:
        for i in range(checkframe+1):
            new_label = nellie_df[nellie_df['reassigned_label_raw'] == label ]
            new_label = new_label[new_label['t']== endingframe-i]
            volume = np.array(list(new_label['organelle_area_raw']))
            r_label = np.array(list(new_label['label']))
            volume_all.append(volume) 
            raw_label.append(r_label)

    # do we need for loop here
    for i in range(checkframe+1):
        volume_value.append(np.sum(volume_all[i]))

    arr = np.array(volume_value[0:checkframe])
    mean = np.average(arr)

    volume_value = np.array(volume_value)
    diff =  volume_value[1] - volume_value[0]
    significance = abs(diff) > percent_threshold #TODO percent threshold = 0.25 but diff is not percent

    #if np.all(volume_value>5):
    #diff= (volume_value[checkframe] - mean) / mean
    #significance = abs(diff) > percent_threshold
    
    # how to prevent false positive -> not fusion but assigned as fusion
    # vary the threshold? 
    # manually collect the data and update threshold. 

    #else:
    #    diff = volume_value[checkframe] - mean
    #    significance = abs(diff)>1

    return volume_all,significance,diff,raw_label

def NN_index_to_label(indices,treeMatrix):
    NN_arr = []
    for index in indices:
        NN = treeMatrix[index,2]
        NN_arr.append(NN)
    NN_arr = np.array(NN_arr)
    
    return(NN_arr)

def check_NN_volume(treeMatrix_pre,treeMatrix_post,dists,idxs,label,frame,isFusion,percent):
    '''
    check the volume and distance of NN
    input: 
        treeMatrix, distance between neighbor and interestd label, index of neighbor in tree, label, frame, bool(isFusion), accepted area
        -> dists and idxs : [N, 8,3] 3 indices of each centriod that closet to 8 extrema. There are in total N centroid from N reassigned labels
    output 
        column
            [0,1] : volume pre and post for fusion/ volume post and pre on fission
            [2] =  smallest distance from extrema to centriod of that label 
            [3] = label  
    
    firstrow = label of intereNNVst. 
    '''
    
    if isFusion:
        treeMatrix = treeMatrix_pre #fusion check the neighbour of the current frame
    else:
        treeMatrix = treeMatrix_post # fission check neighbour of the next frame
    # tree matrix of both frames are included in order to calculate the distance between raw label of frame t and t+1 so we could know which of the component has a fission/fusion
    # allowing us to trace down its location

    distance_arr = []
    temp = np.argwhere(treeMatrix[:,2] == label) # tree indiced where label of tree in this frame is == interested label

    #index of NN 1st column = label index, others = neighbour innex.
    # N closet centriod to current label 8 extremas 
    # distance and indices of all centriod that has label == interested label
    # distance = distance to the 3 closet neighbors
    indices = idxs[temp[:,0]]
    distance = dists[temp[:,0]]


    label_with_corr = treeMatrix[indices.astype(np.dtype(int))] 
    raw_label_filter = label_with_corr[:,:,:,3]
    label_with_corr = label_with_corr.reshape((label_with_corr.shape[0], label_with_corr.shape[1] * label_with_corr.shape[2],  label_with_corr.shape[3] ))
    unique_label = np.unique(label_with_corr, axis = 1)

    reassigned_label_filter = unique_label[:,:,2]
    unique_NN = np.unique(reassigned_label_filter)
    corr = unique_label[:,:,0:2] # centroid of the closet centriod of the neighbor to the extrema of each semantic component

    volume_arr = []
    distance_arr = []
    label_arr = []
    corr_arr = [] # corrdinate of centriod of interested label and its neighbour
    reassigned_label_arr=[]
    i = 0
    for n in unique_NN:
        
        volume,_,_,raw_label = check_volume_value(frame[0],frame[1],n,isFusion,percent_threshold= percent) #index is not a label need to get the label from tree
        #calculate distance between centriod of frame t and t+1 of label N
        rep_len = np.abs(volume[0].shape[0] - volume[1].shape[0]) +1
        filters = ~np.isin(raw_label_filter,raw_label[0])
        dist = []
        min_val = np.min(distance[filters].astype(np.dtype(float)))
        if len(dist) == 0:
            dist = np.array([[min_val]])
        else:
            dist = np.append(dist,min_val)

        mask_loc = np.isin(treeMatrix[:,2],n)
        loc = treeMatrix[mask_loc][:,:2] # coordinate of the interested label and ํits neighbour in frame t

        if not (volume[0].shape[0] == 1 or volume[1].shape[0] == 1) and (volume[1].shape[0] != volume[0].shape[0]):
           # when label has more than 1 component and not equal, need to fill the dist,loc with closet next component
           
           # to query the rawlabel in treeMatrix, change tree according to the fission fusion condition
           # in fusion raw_label[0] match with treeMatrix pre
           # in fission raw_label[0] represent mitochondria after fission at tree t+1 
            if isFusion:
                t_pre = treeMatrix_pre
                t_post = treeMatrix_post
            else:
                t_pre = treeMatrix_post
                t_post = treeMatrix_pre

            mask_t= np.isin(t_pre,raw_label[0]).any(axis=1)
            mask_t1= np.isin(t_post,raw_label[1]).any(axis=1)
            coords1 = t_pre[mask_t][:,:2]
            coords2 = t_post[mask_t1][:,:2]
            delta = coords1[:, np.newaxis, :] - coords2  
            distances = np.sqrt(np.sum(delta**2, axis=2)) # first axis = corr in frame t, second axis = corr in frame t+1
            min_dis = np.sort(np.min(distances, axis=0))

            min_dim = np.min((volume[0].shape[0], volume[1].shape[0]))
            rep_dist = np.repeat(dist, rep_len + min_dim - 1)
            rep_dist = rep_dist[np.newaxis,:]

            for i in range(1,rep_len):
                index_to_repeat = np.where(distances == min_dis[-i])
                if volume[0].shape[0] > volume[1].shape[0]: # number of pre label is more than post label 
                    j = index_to_repeat[1][0] 
                    if i == 1:
                        area = np.array([np.append(subarr, subarr[j]) if len(subarr) == volume[1].shape[0]  else subarr
                                        for subarr in volume], dtype=object)
                        labels = np.array([np.append(subarr, subarr[j]) if len(subarr) == raw_label[1].shape[0]  else subarr
                                        for subarr in raw_label], dtype=object)
                        rep_loc = loc
                    if i > 1:
                        new_area = np.append(area[1], volume[1][j])
                        area[1] = new_area
                        new_label = np.append(labels[1], raw_label[1][j])
                        labels[1] = new_label
                    
                else: # number of post label is more than pre label
                    j = index_to_repeat[0][0] # repeat number of area in t frame
                    if i == 1:
                        rep_loc = np.concatenate((loc,loc[np.newaxis,j,:]),axis = 0)
                        area = np.array([np.append(subarr, subarr[j]) if len(subarr) == volume[0].shape[0]  else subarr
                                        for subarr in volume], dtype=object)
                        labels = np.array([np.append(subarr, subarr[j]) if len(subarr) == raw_label[0].shape[0]  else subarr
                                        for subarr in raw_label], dtype=object)
                    if i > 1 : 
                        new_area = np.append(area[0], volume[0][j])
                        area[0] = new_area
                        new_label = np.append(labels[0], raw_label[0][j])
                        labels[0] = new_label
                        rep_loc = np.append(rep_loc, loc[np.newaxis,j,:], axis=0)


                if area[0].shape[0] == area[1].shape[0]:
                    area = np.vstack(area)
                    labels = np.vstack(labels)


        else:  #
            area = np.array(
                [ np.repeat(subarr, rep_len) if subarr.size == 1 else subarr 
                for subarr in volume], dtype=object)
            labels = np.array(
                [ np.repeat(subarr, rep_len) if subarr.size == 1 else subarr 
                for subarr in raw_label], dtype=object)
            #if labels.shape[1] == 1:
            #    labels = np.transpose(labels)
            
            if (rep_len > 1) and (volume[0].shape[0] != volume[1].shape[0]) and (volume[0].shape[0] != 1):
                # doesnt need to repeat the loc of pre volume
                rep_dist = np.repeat(dist, rep_len)
                rep_dist = rep_dist[np.newaxis,:]
                rep_loc = loc
                #rep_loc = np.repeat(loc, rep_len, axis=0)

            elif (rep_len > 1) and (volume[0].shape[0] != volume[1].shape[0]) and (volume[0].shape[0] == 1):
                # loc is repeated since the number of component in pre is 1 (loc is coordiates of pre volume)
                rep_dist = np.repeat(dist, rep_len)
                rep_dist = rep_dist[np.newaxis,:]
                rep_loc = np.repeat(loc, rep_len, axis=0)

            elif (volume[0].shape[0] == volume[1].shape[0]) and (volume[0].shape[0] != 1):
                rep_dist = np.repeat(dist, volume[0].shape[0])
                rep_dist = rep_dist[np.newaxis,:]
                rep_loc = loc
               # rep_loc = np.repeat(loc, volume[0].shape[0], axis=0)
            else: 
                rep_dist = dist
                rep_loc = loc

        reassigned_label = np.repeat(n,labels.shape[1])[np.newaxis,:]
        i+=1
        if len(volume_arr) == 0:
            volume_arr = area
            distance_arr = rep_dist 
            label_arr = labels
            corr_arr = rep_loc
            reassigned_label_arr = reassigned_label
        else:
            volume_arr = np.concatenate((volume_arr,area),axis = 1)
            distance_arr = np.concatenate((distance_arr,rep_dist),axis = 1)
            label_arr = np.concatenate((label_arr,labels),axis = 1)
            corr_arr = np.concatenate((corr_arr,rep_loc),axis = 0)
            reassigned_label_arr = np.concatenate((reassigned_label_arr,reassigned_label),axis = 1)

    volume_arr = np.transpose(volume_arr)
    distance_arr = np.transpose(distance_arr)
    label_arr = np.transpose(label_arr)
    reassigned_label_arr = np.transpose(reassigned_label_arr)
    
    final_arr = np.concatenate((volume_arr,distance_arr,reassigned_label_arr, label_arr,corr_arr),axis = 1)
    sorted_indices = np.argsort(final_arr[:, 0])
    final_arr = final_arr[sorted_indices]
    final_arr = move_row_to_first(final_arr, label)
    final_arr = pd.DataFrame(final_arr, columns=['area_t', 'area_t+1', 'dist_to_N',  'reassigned_label', 'raw_label_t','raw_label_t+1', 'y_corr', 'x_corr'])

    # include location 
    return final_arr
    
def find_combinations(arr,error_percentage,start=1,):
    '''
    find a combination of first row that could potentially sum up to the target value in second row
    return value pair data of
        1. a possible pair data of data_column to the specific_cell_value (result data)
        2. a possible pari data of target_column to the specific_cell_value (result_target)
    
        Equation:
        specific_cell_value + pre_event_arr = target_col + result_target
    '''
    pre_event_arr = []
    post_event_arr = []
    result_label = []
    result_dists = []
    data_col = arr[:, 0]
    target_col = arr[:, 1]
    dists_col = arr[:,2]
    label_col = arr[:,3]

    target_value = arr[0, 1]
    specific_cell_value = arr[0, 0]
    remaining_target = target_value - specific_cell_value
    sum_all_vol = target_value+specific_cell_value

    upper_boundary, lower_boundary = error_percentage*target_value, -error_percentage*target_value

    def backtrack(data_list,target_list,label_list,dist_list, start, remaining,sum,lower_bound,upper_bound):

        if (remaining/ (sum/2)) <= error_percentage and len(data_list) > 0 :
        #remaining >= lower_bound and remaining <= upper_bound :
            #remaining inbetween this range == acceptable
            pre_event_arr.append(data_list[:])
            post_event_arr.append(target_list[:])
            result_label.append(label_list[:])
            result_dists.append(dist_list[:])
            return
        
        for i in range(start, len(data_col)):
            #second condition after and allow it to select area that only decrease
            if target_col[i] < data_col[i]:#remaining - data_col[i] >= lower_boundary and 
                data_list.append(data_col[i])
                target_list.append(target_col[i])
                label_list.append(label_col[i])
                dist_list.append(dists_col[i])
                '''
                #Above condition
                diff = (remaining - data_col[i] + target_col[i])/ (sum_all_vol/2)
                diff <= 0.15 #exit 
                #need to check how good it is 

                #for condition below
                sum_all_vol + data_col[i] + target_col[i]
                '''

                backtrack(data_list,target_list,label_list,dist_list, i + 1, remaining - (data_col[i]-target_col[i]),
                        sum + data_col[i] + target_col[i],lower_boundary, upper_boundary)
                data_list.pop()
                target_list.pop()
                label_list.pop()
                dist_list.pop()
    #when the lable suddenly appear or disappear = specific_cell_volume = 0 -> break the combination

    if specific_cell_value == 0: 
        return np.array([])

    else:
        backtrack([],[],[], [],start, remaining_target,sum_all_vol ,lower_boundary,upper_boundary)
        volume_pre = np.array(pre_event_arr,dtype=object)
        volume_post = np.array(post_event_arr,dtype=object)
        label = np.array(result_label,dtype=object)
        dist = np.array(result_dists,dtype=object)

    result = np.transpose(np.concatenate((volume_pre,volume_post,dist,label)))
    return result
    
def gaussian_distance_weight(distance,sigma = 1.0):
    distance = distance.astype(float)
    return np.exp(-distance**2 / 2*sigma**2)

def inverse_distance_weight(distance,ratio):
    return  1/distance**ratio

def exponential_decay(distance,gamma):
    distance = distance.astype(float)
    return np.exp(-gamma*distance)

def runframe(nearest_N,isFusion,label,interested_frame,fus_fiss_arr, percent,error): 
#runframe(treeMatrix_pre, treeMatrix_post,dists_tree,idxs,isFusion,label,interested_frame,fus_fiss_arr, percent,error):

    '''
        run fission fusion candidate in each frame with all possible combination calculation
        input: 
            -> treeMatrix, distance between neighbor and interestd label, index of neighbor in tree, label, frame, bool(isFusion), accepted area

        output:
            -> unique_arr = [pre_event_volume,post_event_volume,distance between label and nearest neighbour,label]
    '''
    area_neighbours = nearest_N.iloc[:,0:4].to_numpy()
    mask = np.isin(area_neighbours[:,3],label)
    check_num = area_neighbours[mask].astype(float)
    comp_before = np.unique(check_num[:,0],axis = 0).shape[0]
    comp_after = np.unique(check_num[:,1],axis = 0).shape[0]

    if comp_before == comp_after:

        combinations = find_combinations(area_neighbours, error_percentage = error)

        if combinations.size == 0:
            return None

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
            
        except: #never execute here in simulation code. 
            combinations = np.concatenate(combinations)
            section_index = int(len(combinations)/4)
            pre_event_volume =  np.concatenate(np.expand_dims(combinations[:section_index],axis=0))
            post_event_volume = np.concatenate(np.expand_dims(combinations[section_index:2*section_index],axis=0))
            dists = np.concatenate(np.expand_dims(combinations[2*section_index:3*section_index],axis=0))
            label_arr = np.concatenate(np.expand_dims(combinations[3*section_index:4*section_index],axis=0))
        
        # mask use to find exact coordinate of the label since label can be repeated.
        # area of pre and post event mitochondria are unique. 
        mask_loc_pre = np.isin(nearest_N['area_t'],pre_event_volume)
        mask_loc_post = np.isin(nearest_N['area_t+1'],post_event_volume)
        mask_loc = mask_loc_pre & mask_loc_post

        coor_y =  nearest_N[mask_loc].iloc[:,6:7].to_numpy().transpose().astype(np.float64)
        coor_x =  nearest_N[mask_loc].iloc[:,7:8].to_numpy().transpose().astype(np.float64)

        arr = np.concatenate((np.expand_dims(pre_event_volume,axis = 1),
                              np.expand_dims(post_event_volume,axis = 1),
                              np.expand_dims(dists,axis = 1),
                              np.expand_dims(label_arr,axis = 1), 
                              np.expand_dims(coor_x,axis = 1),
                              np.expand_dims(coor_y,axis = 1)), 
                              axis = 1,dtype=np.float64)
        unique_arr = np.unique(arr,axis=0)

        if unique_arr.ndim == 3:
            unique_arr = unique_arr.transpose(1,0,2).reshape(6,-1).T

        if ~isFusion:
            unique_arr[:,[0,1]] = unique_arr[:,[1,0]]
            
        weight2 = inverse_distance_weight(unique_arr[:,2],2)
        #weight = gaussian_distance_weight(unique_arr[:,0])
        #weight3 = exponential_decay(unique_arr[:,0],1)
        all_weight = np.sum(weight2)
        normalized_weight = weight2/all_weight
        event_prob = np.max(normalized_weight)

        #prob == 1 when any of the two closet neighbor ==0
        if np.any(unique_arr[:2,1] == 0):
            fus_fiss_arr[label-1][interested_frame[1]] += 1
        else:
            fus_fiss_arr[label-1][interested_frame[1]] += event_prob

        return unique_arr

    else:
        mask = np.isin(nearest_N['reassigned_label'],label)
        _, idx = np.unique(nearest_N[mask]['x_corr'].to_numpy().astype(np.float64), return_index=True)
        coor_x = nearest_N[mask]['x_corr'].to_numpy().astype(np.float64)[np.sort(idx)]

        _, idx = np.unique(nearest_N[mask]['y_corr'].to_numpy().astype(np.float64), return_index=True)
        coor_y = nearest_N[mask]['y_corr'].to_numpy().astype(np.float64)[np.sort(idx)]

        coor_x = np.expand_dims(coor_x, axis =1 )
        coor_y = np.expand_dims(coor_y, axis = 1)
        coor_all = np.concatenate((coor_x,coor_y),axis = 1)

        unique_arr = np.array([np.append(np.zeros(4), arr) for arr in coor_all])

        fus_fiss_arr[label-1][interested_frame[1]] += np.abs(comp_before-comp_after)
        return unique_arr

def append_NN(NN,labels,frame,neighbor_arr,isFusion):
    label_NN = np.full(shape=(NN.shape[0],1) , fill_value = labels)
    
    fusion_arr = np.full(shape=(NN.shape[0],1) , fill_value = isFusion)
    if isFusion:
        frame_NN = np.full(shape=(NN.shape[0],1) , fill_value = frame)
    else:
        frame_NN = np.full(shape=(NN.shape[0],1) , fill_value = frame+1)

    NN_volume = np.concatenate((NN,fusion_arr,frame_NN,label_NN),axis = 1)
    if len(neighbor_arr) == 0:
        neighbor_arr = np.array(NN_volume)
    else:
        neighbor_arr = np.concatenate((neighbor_arr,NN_volume),axis =0) 
    return neighbor_arr

def append_event(event,diff,labels,frame,isFusion,event_arr):
    fusion_arr = np.full(shape=(event.shape[0],1) , fill_value = isFusion)
    label_NN = np.full(shape=(event.shape[0],1) , fill_value = labels)
    diff = np.full(shape=(event.shape[0],1) , fill_value = diff)

    if isFusion:
        frame_NN = np.full(shape=(event.shape[0],1) , fill_value = frame)
    else:
        frame_NN = np.full(shape=(event.shape[0],1) , fill_value = frame+1)
        
    fis_fus = np.concatenate((event,diff,fusion_arr,frame_NN,label_NN),axis = 1)

    if len(event_all) == 0:
        event_arr = np.array(fis_fus)
    else:
        event_arr = np.concatenate((event_arr,fis_fus),axis =0) 
    return event_arr

# 4 *21 = 84 = 1.5 hrs

if __name__ == "__main__":
    main_dir = "/home/jirapong/jirapong/latest_network_data_summary/nellie_output/"
    out_path = "/home/jirapong/nellie/my_script/nellie_output/simulated/new_simulation/"
    #filename = "Rotenone"
    diff_threshold = 0.25
    combination_threshold = 23

    '''
    file_path_feature = main_dir  + "ins1_" + filename +  ".ome-ch0-features_components.csv"
    seg_path = main_dir + "ins1_" + filename  + ".ome-ch0-im_instance_label.ome.tif"
    reassigned_path = main_dir+ "ins1_" + filename  + ".ome-ch0-im_obj_label_reassigned.ome.tif"
    
    
    file_path_feature = main_dir  + "time_ins_" + filename +  ".ome-ch0-features_components.csv"
    seg_path = main_dir + "time_ins_" + filename  + ".ome-ch0-im_instance_label.ome.tif"
    reassigned_path = main_dir+ "time_ins_" + filename  + ".ome-ch0-im_obj_label_reassigned.ome.tif"
    '''
    fission_path_csv = "branch_binary.ome-TYX-T1p0_Y0p2_X0p2-ch0-t0_to_100-features_organelles.csv"
    #fusion_path_csv = "hi_fu_thick.ome-TYX-T1p0_Y0p1_X0p1-ch0-t50_to_100-features_organelles.csv"
    fission_path_image = "branch_binary.ome-TYX-T1p0_Y0p2_X0p2-ch0-t0_to_100-im_instance_label.ome.tif"
    #fusion_path_image = "hi_fu_thick.ome-TYX-T1p0_Y0p1_X0p1-ch0-t50_to_100-im_instance_label.ome.tif"

    file_path_feature = main_dir  + fission_path_csv # "hi_fu_thick.ome-TYX-T1p0_Y0p1_X0p1-ch0-t50_to_100-features_organelles.csv"
    seg_path = main_dir + "nellie_necessities/" + fission_path_image #hi_fi_thick.ome-TYX-T1p0_Y0p1_X0p1-ch0-t50_to_100-im_instance_label.ome.tif
    #hi_fu_thick.ome-TYX-T1p0_Y0p1_X0p1-ch0-t50_to_100-im_instance_label.ome.tif"
    

    output_name = "simulated_normal"

    nellie_df = pd.read_csv(file_path_feature)
    labeled_im = tifffile.imread(seg_path)
    #reassigned_im = tifffile.imread(reassigned_path)

    first_frame_label =  len(nellie_df['reassigned_label_raw'].unique())
    max_frame_num = int(nellie_df['t'].max()) + 1

    final_fission_self = np.zeros((first_frame_label,max_frame_num))
    final_fusion_self = np.zeros((first_frame_label,max_frame_num))
    final_fission_volume = np.zeros((first_frame_label,max_frame_num))
    final_fusion_volume = np.zeros((first_frame_label,max_frame_num))

    final_fusion_label= np.zeros((first_frame_label,max_frame_num),dtype=object)
    final_fission_label = np.zeros((first_frame_label,max_frame_num),dtype=object)
    fragment = 0
    fission = np.zeros(max_frame_num)
    fusion = np.zeros(max_frame_num)

    treeMatrix_all = []
    dists_all = []
    idxs_all = []
    tree_all = []
    neighbour_all = []
    event_all = []

    NN_all = []

    frame_all = []
    volume_all = []
    sig_all = []
    label_all = []
    for labels in tqdm(range(1,first_frame_label)):
        final_fission_self,final_fusion_self = check_mito_number(nellie_df, labels,final_fission_self,final_fusion_self)

        for frame in range(max_frame_num):
            # what if there's no label == 1 in some frame? no nearest neighbour?
            # no. Since we run it through all label. label ==1 would be the first one we run though regardless, so it mean we can query all 
            # nearest neighbour of the image in the first frame.

            area_in_frame = nellie_df[nellie_df['t'] == frame]['organelle_area_raw']
            frame_ = nellie_df[nellie_df['t'] == frame]
            current_label_area = np.sum(frame_[frame_['reassigned_label_raw'] == labels]['organelle_area_raw'])

            avg_area = np.mean(area_in_frame)
            std = np.std(area_in_frame)
            accepted_area = avg_area #+ std # 1sd is better from the observation

            if labels == 1: #in the first run of each frame, we query treeMatrix of all label. 
                # query tree OF 1 frame ahead
                if len(treeMatrix_all) == 0 and frame < max_frame_num - 2:
                    treeMatrix,dists,idxs = nearest_neighbour(labeled_im,nellie_df,frame=frame)
                    treeMatrix_post,dists_post,idxs_post = nearest_neighbour(labeled_im,nellie_df,frame=frame + 1)
                    treeMatrix_all.append(treeMatrix)
                    treeMatrix_all.append(treeMatrix_post)
                    dists_all.append(dists)
                    dists_all.append(dists_post)
                    idxs_all.append(idxs)
                    idxs_all.append(idxs_post)
                elif frame < max_frame_num - 2:
                    treeMatrix,dists,idxs = nearest_neighbour(labeled_im,nellie_df,frame=frame +1)
                    treeMatrix_all.append(treeMatrix)
                    dists_all.append(dists)
                    idxs_all.append(idxs)

            if frame < max_frame_num-2 : #and current_label_area > accepted_area: # uncomment this for actual mito image

                volume,significance,diff,_ = check_volume_value(frame,frame +1 ,labels,isFusion = True, percent_threshold= diff_threshold )
                interested_frame = np.array([frame,frame+1])
                volume_all.append(volume)
                frame_all.append(frame)
                label_all.append(labels)
                sig_all.append(significance)

                # to output all neighbour. we might skip neighbor output if it is inside the area and combination criterion.
                if diff >= 0: 
                    isFusion_condition = True
                    neigh = check_NN_volume(treeMatrix_all[frame], treeMatrix_all[frame +1],dists_all[frame],idxs_all[frame],
                                            label= labels, frame = interested_frame, isFusion= isFusion_condition, percent = diff_threshold)
                elif diff < 0:
                    isFusion_condition = False
                    neigh = check_NN_volume(treeMatrix_all[frame], treeMatrix_all[frame +1],dists_all[frame+1],idxs_all[frame+1],
                                            label= labels, frame = interested_frame, isFusion= isFusion_condition, percent = diff_threshold)
                if neigh is not None: 
                    NN_all = append_NN(neigh,labels,frame,NN_all,isFusion = True)

                isAreaNotZero = np.all([np.all(subarray) for subarray in volume])
                
                if significance and isAreaNotZero:
                    event_arr  = runframe(neigh, isFusion_condition,labels,interested_frame,final_fusion_volume, percent= diff_threshold,error=combination_threshold)

                    if event_arr is not None and not isFusion_condition : 
                        # combination of event calculate only increaase in area. So the combination function swap area column (area at t+1, area at t) to make it as if area is increasing
                        # after combination is calculate, the column is reverse (area at t, area at t+1)
                        event_arr[:,[0,1]] = event_arr[:,[1,0]]
                            
                    if event_arr is not None and event_arr.size >0 :
                        event_all = append_event(event_arr,diff, labels,frame,isFusion_condition,event_all)

    column_names_Neighbour = ["Volume_pre", "Volume_post", "Distance","reassigned_label","raw_label_t","raw_label_t+1",
                              "y_coor","x_coor","isFusion" ,"Frame" , "Label"]
    label_NN_all = pd.DataFrame(NN_all, columns=column_names_Neighbour)
    #all_volume =  np.concatenate((np.array(volume_all),
                                #np.expand_dims(np.array(sig_all),axis = 1),
                               # np.expand_dims(np.array(label_all),axis = 1),
                               # np.expand_dims(np.array(frame_all),axis = 1)),
                                 #   axis = 1 )

    #column_volume_check = ["Volume_pre", "Volume_post","Significance","label", "Frame"]
    #volume_check_csv = pd.DataFrame(all_volume, columns=column_volume_check)

    column_names_event = ["Volume_pre", "Volume_post", "Distance","Nearest_Label","y_corr", "x_corr", "Volume_diff","isFusion" ,"Frame" , "Label"]
    possible_event_all = pd.DataFrame(event_all, columns=column_names_event)



    try:
        label_NN_all.to_csv(os.path.join(out_path,f'{output_name}_neighbour.csv'), index=False) 
        #volume_check_csv.to_csv(os.path.join(out_path,f'{output_name}_volume_Check.csv'), index=False) 
        possible_event_all.to_csv(os.path.join(out_path,f'{output_name}_event.csv'), index=False) 
        np.savetxt(os.path.join(out_path,f'{output_name}_output_fission.csv'),final_fission_volume, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(out_path,f'{output_name}_output_fusion.csv'),final_fusion_volume, delimiter=',', fmt='%f')

        #volume_check_csv.to_csv(os.path.join(out_path,f'{file[:-10]}_anotated.csv'), index=False) 
    except OSError:
        os.mkdir(os.path.join(out_path)) 
        label_NN_all.to_csv(os.path.join(out_path,f'{output_name}_neighbour.csv'), index=False) 
        #volume_check_csv.to_csv(os.path.join(out_path,f'{output_name}_volume_Check.csv'), index=False) 
        possible_event_all.to_csv(os.path.join(out_path,f'{output_name}_event.csv'), index=False) 
        np.savetxt(os.path.join(out_path,f'{output_name}_output_fission.csv'),final_fission_volume, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(out_path,f'{output_name}_output_fusion.csv'),final_fusion_volume, delimiter=',', fmt='%f')




