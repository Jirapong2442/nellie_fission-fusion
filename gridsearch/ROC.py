import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
def plot_multiple_roc(fpr_list, tpr_list, plot_label_list, names, title="Multiple ROC Curves (K = 4)"):
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
    plt.figure(figsize=(12, 8))
    
    # Plot each ROC curve
    for i, (fpr, tpr, labels, name, color) in enumerate(zip(fpr_list, tpr_list, plot_label_list, names, colors)):
        # Plot connected lines
        plt.plot(fpr, tpr, color=color, alpha=0.5, label=f'{name} (Path)')
        
        # Plot points
        plt.scatter(fpr, tpr, color=color, s=100, alpha=0.6, label=f'{name} (Points)')
        
        # Add labels for points
        
        for j, label in enumerate(labels):
            plt.annotate(f'',
                        (fpr[j], tpr[j]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        color=color,
                        alpha=0.7)
        
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    
    # Add grid, labels, and title
    plt.grid(True, alpha=0.3)
    plt.xlabel('False Positive Rate (FPR)', fontsize=18)
    plt.ylabel('True Positive Rate (TPR)', fontsize=18)
    plt.title(title, fontsize = 18)
    
    # Adjust legend to remove duplicate point entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    seen_names = set()
    
    for handle, label in zip(handles, labels):
        name = label.split(' (')[0]
        if name not in seen_names:
            seen_names.add(name)
            unique_labels.append(name)
            unique_handles.append(handle)
    
    plt.legend(unique_handles, unique_labels, loc='lower right', title="M in percentage")
    # Display the plot
    plt.show()

def get_performace_matrix(dir_path,):

    result_df = []
    name_df = []
    folders = os.listdir(dir_path)
    for folder in folders:
        files = os.listdir(os.path.join(dir_path,folder))
        #for file in files:
            #if "diff" in file:
        file = files[-1]
        value = np.load(os.path.join(dir_path,folder,file))
        value = np.sum(value,axis = 0)

        
        if len(result_df) == 0:
            result_df.append(value)
            result_df = np.array(result_df)
            name_df.append(folder)
            name_df = np.array(name_df)
        else:
            result_df = np.vstack((result_df,value))
            name_df = np.vstack((name_df,folder))
                #plot coordinate name
    return result_df,name_df

if __name__ == "__main__":
    all_tpr = []
    all_fpr = []

    all_label = []

    all_comb = ['0','5','10','15','17','19','21','23','25','50','75','100']
    #all_comb = ['23']
    
    for comb in all_comb:
        #comb = "15"
        dir_path_n3_10s = "D:/Internship/NTU/data_for_script/gridsearch10s/test_neighbor_" + comb + "/"
        dir_path_n2_10s = "D:/Internship/NTU/data_for_script/gridsearch10s_n2/test_neighbor_" + comb + "/"
        dir_path_n4_10s = "D:/Internship/NTU/data_for_script/gridsearch10s_n4/test_neighbor_" + comb + "/"

        dir_path_n3 = "D:/Internship/NTU/data_for_script/gridsearch/test_neighbor_" + comb + "/"
        dir_path_n2 = "D:/Internship/NTU/data_for_script/gridsearchn2/test_neighbor_" + comb + "/"
        dir_path_n4 = "D:/Internship/NTU/data_for_script/gridsearchn4/test_neighbor_" + comb + "/"
        outpath = "./gridsearch/metrics_toxin/n4/"

        all_arr = []
        result = []
        name = []
        
        result_n3_10s,name_n3_10s = get_performace_matrix(dir_path_n3_10s)
        result_n2_10s,name_n2_10s = get_performace_matrix(dir_path_n2_10s)
        result_n4_10s,name_n4_10s = get_performace_matrix(dir_path_n4_10s)


        result_n3,name_n3 = get_performace_matrix(dir_path_n3)
        result_n2,name_n2 = get_performace_matrix(dir_path_n2)
        result_n4,name_n4 = get_performace_matrix(dir_path_n4)

        result_n2_all = result_n2_10s + result_n2
        result_n3_all = result_n3_10s + result_n3
        result_n4_all = result_n4_10s + result_n4

        #if comb == '15':
            # for 15 comb to sort the file name
        #    for i in range(int(len(folders)/4)):
        #        if i == 0:
        #            folders = [folders[(i+1)*3 + i]] + folders[4*i: 4*(i) + 3] + folders[4* (i+1):]
        #        else:
        #            folders = folders[0:4*i]+ [folders[(i+1)*3 + i]] + folders[4*i: 4*(i) + 3] + folders[4* (i+1):]
 


        plot_label = []
        area_thresh = []
        for n in name_n2_10s:
            tmp = []
            area = []
            x = n[0]
            if len(x) == 16 or len(x) == 15:
                tmp = x[5:9] + x[-1]
                area = x[5:9]

            elif len(x) == 17:
                tmp = x[5:9] + x[-2:]
                area = x[5:9]

            elif len(x) ==18:
                tmp = x[5:9] + x[-3:]
                area = x[5:9]

            plot_label.append(tmp)
            area_thresh.append(area)

        all_label.append(plot_label)
        print(comb)
        #print(np.sum(result,axis = 0)) 
        #TP delete the last row (last = all negative = should be 0)
        #result[:,0] = result[:,0] - result[-1,0]

        #TN delete first row = fist row = all positive = TN should be 0
        #result[:,1] = result[:,1] - result[0,1]

        # FP delete the last row (last = all negative = should be 0)
        #result[:,2] = result[:,2] - result[-1,2]

        # FN delete first row = fist row = all positive = FN should be 0
        #result[:,3] = result[:,3] - result[0,3]
        result = result_n4_all

        # TP TN FP FN
        tpr = result[:,0]/(result[:,0] + result[:,3]) #TP/(TP+FN) recall
        fpr = result[:,2]/(result[:,2] + result[:,1]) 
        
        accuracy = (result[:,0] + result[:,1]) / (result[:,0] + result[:,1]  + result[:,2] + result[:,3] )
        precision = result[:,0]/(result[:,0] + result[:,2])
        recall =tpr
        f1_score = (2 * precision * recall) / (precision + recall)
        # add 0,0 and 1,1
        tpr_extended = np.concatenate([[1],tpr,[0]])
        fpr_extended = np.concatenate([[1],fpr,[0]])

        auc = np.trapz(tpr_extended,x = fpr_extended)
        print(auc)

        all_tpr.append(tpr_extended)
        all_fpr.append(fpr_extended)  

        #which is more common 1. store everythin in the same file different label, store every in different accessible file
        #save file
        
        all_arr = np.concatenate((np.expand_dims((np.repeat(comb,len(tpr))),axis = 1), #remove the 1 and 0 at the beginninng and the end
                                    np.reshape(np.array(area_thresh),(21,1)),
                                    np.expand_dims((np.repeat('3',len(tpr))),axis = 1),
                                    np.expand_dims(np.array(tpr),axis = 1),
                                    np.expand_dims(np.array(fpr),axis = 1),
                                    np.expand_dims((np.repeat(auc,len(tpr))),axis = 1),
                                    np.reshape(np.array(accuracy),(21,1)),
                                    np.reshape(np.array(precision),(21,1)),
                                    np.reshape(np.array(recall),(21,1)),
                                    np.reshape(np.array(f1_score),(21,1))),
                                    axis = 1 )
        column_names_event = ["combination_threshold" , "area_threshold" , "num_neighbor","tpr", "fpr", "auc", "accuracy", "precesion","recall","f1_score"]
        possible_event_all = pd.DataFrame(all_arr, columns=column_names_event)
        possible_event_all.to_csv(os.path.join(outpath,f'{comb}_metrics.csv'), index=False) 

    plot_multiple_roc(all_fpr, all_tpr, all_label, all_comb, "Comparison of ROC Curves (K=4)")
    