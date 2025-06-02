import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_multiple_line(fpr_list, tpr_list, plot_label_list, names, title="Multiple ROC Curves" ):
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
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of frame')
    plt.ylabel('number of event')
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
    
    plt.legend(unique_handles, unique_labels, loc='upper right')
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    all_label = []
    all_area = []
    all_index = []
    all_raw_label = []
    label_path = "./check_label&area/toxicity/"

    files= os.listdir(label_path)
    all_name = [files[i][0:5] for i in range(len(files))]
    #all_name = ['control 10min' , 'control' , 'mdivi10min','mdivi']

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

    plot_multiple_line(all_index, all_raw_label, all_index , all_name, "Comparison of label")
    # used index: all_index[2:6], all_area[2:6], all_index[2:6] , all_name[2:6]
    '''
    plt.boxplot([np.ravel(all_area[0]),np.ravel(all_area[1]),np.ravel(all_area[2]),np.ravel(all_area[3]), 
                np.ravel(all_area[4]), np.ravel(all_area[5]),np.ravel(all_area[6]), np.ravel(all_area[7])], labels=all_name)
    # Adding a title and labels
    plt.title('summation of area of different label')
    plt.ylabel('Values')

    # Display the plot
    plt.show()
    '''