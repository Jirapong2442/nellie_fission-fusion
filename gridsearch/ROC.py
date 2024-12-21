import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
def plot_multiple_roc(fpr_list, tpr_list, plot_label_list, names, title="Multiple ROC Curves"):
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
            plt.annotate(f'{label}',
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
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    
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
    
    plt.legend(unique_handles, unique_labels, loc='lower right')
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    all_tpr = []
    all_fpr = []
    all_label = []
    #all_comb = ['0','5','10','15','17','19','21','23','25','31','50','75','100']
    all_comb = ['23']

    for comb in all_comb:
        #comb = "15"
        dir_path = "./gridsearch/test_neighbor_" + comb + "/"
        folders = os.listdir(dir_path)
        result = []
        name = []

        if comb == '15':
            # for 15 comb to sort the file name
            for i in range(int(len(folders)/4)):
                if i == 0:
                    folders = [folders[(i+1)*3 + i]] + folders[4*i: 4*(i) + 3] + folders[4* (i+1):]
                else:
                    folders = folders[0:4*i]+ [folders[(i+1)*3 + i]] + folders[4*i: 4*(i) + 3] + folders[4* (i+1):]
 
        for folder in folders:
            files = os.listdir(os.path.join(dir_path,folder))
            #for file in files:
                #if "diff" in file:
            file = files[-1]
            value = np.load(os.path.join(dir_path,folder,file))
            value = np.sum(value,axis = 0)

            
            if len(result) == 0:
                result.append(value)
                result = np.array(result)
                name.append(file)
                name = np.array(name)
            else:
                result = np.vstack((result,value))
                name = np.vstack((name,file))
        print(comb)
        #print(np.sum(result,axis = 0)) 
        
        tpr = result[:,0]/(result[:,0] + result[:,3]) #TP/(TP+FN)
        fpr = result[:,2]/(result[:,2] + result[:,1]) #TP/(TP+FN)
        
        # add 0,0 and 1,1
        tpr = np.concatenate([[1],tpr,[0]])
        fpr = np.concatenate([[1],fpr,[0]])

        all_tpr.append(tpr)
        all_fpr.append(fpr)       

        # TP TN FP FN
        # more combination threshold = more positive = approching 11.
        # less == approacing  00 
        auc = np.trapz(tpr,x = fpr)
        print(auc)

        #plot coordinate name
        plot_label = []
        for n in name:
            tmp = []
            x = n[0]
            if len(x) == 18:
                tmp = x[5:7] + x[12:14]

            elif len(x) == 21:
                tmp = x[5:9] + x[14:17]

            elif len(x) == 20:
                tmp = x[5:8] + x[13:16]

            plot_label.append(tmp)

        all_label.append(plot_label)

    plot_multiple_roc(all_fpr, all_tpr, all_label, all_comb, "Comparison of ROC Curves")
    