import numpy as np
import pandas as pd
import os


path = "./all/15_10/"
annotation_path = "./template/"
labels = [80,323,200,345,26,255, 78,296, 250,304,198,321,31,249,268,301,] # 10 0 first
#labels = [ 78,296, 26,255, 250,304,198,321,31,249,268,301, 80,323,200,345,] # 1 0 first
index = 0

def predict(check_label,mini_df):
    working_arr = mini_df[mini_df["Label"] == check_label]
    working_arr = np.array(working_arr)

    working_arr[working_arr[:, 2] == 0, 0] -= 1
    output = np.unique(working_arr[:,0])
    return output

def annotate (check_label,output):
    df = annotation[annotation["Label"] == check_label]
    df = np.array(df)
    df[:,8] = 0

    for x in output:
        df[df[:, 6] == x, 8] = 1
    return df

for file in os.listdir(path):
    if "event" in file:
        event = pd.read_csv(os.path.join(path,file))
        mini_df = event[["Frame","Label","isFusion"]]
        output1 = predict(labels[index],mini_df)
        output2 = predict(labels[index+1],mini_df)
        
    if file[:-10] + "glu.xlsx" in os.listdir(annotation_path): 
        annotation = pd.read_excel(os.path.join(annotation_path,file[:-10] + "glu.xlsx"),sheet_name = "Sheet1") 
        column = annotation.columns
        df_1 = annotate(labels[index],output1)
        df_2 = annotate(labels[index+1],output2)
        
        output_annotate = np.vstack((df_1,df_2))
        volume_check_csv = pd.DataFrame(output_annotate, columns= column)
        volume_check_csv.to_csv(f'{file[:-10]}_anotated.csv', index=False) 
        index += 2

'''
event = pd.read_csv(path + file + ".csv") 
mini_df = event[["Frame","Label","isFusion"]]
annotation = pd.read_excel(annotation_path,sheet_name = "Sheet1") 
column = annotation.columns

check_label_1 = 80
check_label_2 = 323


output1 = predict(check_label_1)
output2 = predict(check_label_2)

df_1 = annotate(check_label_1,output1)
df_2 = annotate(check_label_2,output2)

output_annotate = np.vstack((df_1,df_2))
volume_check_csv = pd.DataFrame(output_annotate, columns= column)
volume_check_csv.to_csv(f'{file}_anotated.csv', index=False) 

'''