import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

main_dir = "D:/Internship/NTU/nellie_output/start 1"
file_path_feature2 = main_dir + "/1_resized_2_glu_1min_#1.ome-ch0-features_components.csv"
file_path_feature1 = main_dir + "/1_resized_1_glu_1min_#1.ome-ch0-features_components.csv"
file_path_feature0 = main_dir + "/1_resized_0_glu_1min_#1.ome-ch0-features_components.csv"
file_path_feature05 = main_dir + "/1_resized_0.5_glu_1min_#1.ome-ch0-features_components.csv"

df2 = pd.read_csv(file_path_feature2)
df1 = pd.read_csv(file_path_feature1)
df0 = pd.read_csv(file_path_feature0)
df05 = pd.read_csv(file_path_feature05)

mini_df2 = df2[['reassigned_label_raw','t','area_raw'] ]
mini_df1 = df1[['reassigned_label_raw','t','area_raw'] ]
mini_df05 = df05[['reassigned_label_raw','t','area_raw'] ]
mini_df0 = df0[['reassigned_label_raw','t','area_raw'] ]

area = np.concatenate((np.array(mini_df0['area_raw']),np.array(mini_df05['area_raw']),np.array(mini_df1['area_raw']),np.array(mini_df2['area_raw'])),axis = 0)

area = area[~np.isclose(area, 0)]

median_value = np.median(area)
mean_value = np.mean(area)

bin = np.arange(0, 20.5, 0.5)  # 20.5 to include 20

# Create the plot
plt.figure(figsize=(12, 6))

# Plot histogram
plt.hist(area, bins=bin, edgecolor='black', alpha=0.7, label='Data')

# Add vertical line for median
plt.axvline(x=median_value, color='red', linestyle='--', 
            label=f'Median = {median_value:.2f}')
plt.axvline(x=mean_value, color='green', linestyle='--', 
            label=f'Mean = {mean_value:.2f}')

# Customize the plot
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Area Histogram of 0,0.5,1,2 glucose concentration')
plt.grid(True, alpha=0.3)
plt.legend()
ymax = plt.ylim()[1]
# Optional: Annotate the median with text
plt.annotate(f'Median: {median_value:.2f}', 
            xy=(median_value, plt.ylim()[1]),
            xytext=(10, 10), textcoords='offset points',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->'))
plt.annotate(f'Mean: {mean_value:.2f}', 
            xy=(mean_value, ymax*0.9),  # Slightly lower than median annotation
            xytext=(10, -10), textcoords='offset points',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5),
            arrowprops=dict(arrowstyle = '->'))

plt.show()