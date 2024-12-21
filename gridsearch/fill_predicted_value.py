import numpy as np
import pandas as pd
import os

diff_threshold = ['0.0','0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0',
                  '2.25','2.5','2.75','3.0','3.25','3.5','3.75','4.0','4.25',
                  '4.5','4.75','5.0']

comb_threshold = ['0', '5', '10', '15', '17', '19' , '21', '23','25', '31', '50', '75', '100'] #,
# done '17','19','21','23','25','27','31'
for i in range(len(comb_threshold)):
    comb = comb_threshold[i]
    output_folder = './gridsearchn2/test_neighbor_' + comb
    os.mkdir(output_folder) 

    for j in range(len(diff_threshold)):
        diff = diff_threshold[j]
        path = "./all_n2/" + diff + "_" + comb + "/"
        annotation_path = "./template/"
        output_path =  output_folder +'/diff_' + diff + "_comb_" + comb
        #labels = [80,323,200,345,26,255, 78,296, 250,304,198,321,31,249,268,301,] # 10 0 first
        labels = [ 78,296, 26,255, 250,304,198,321, 
                  268,301,31,249, 80,323,200,345,] # 1 0 first
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
                try: 
                    volume_check_csv.to_csv(os.path.join(output_path,f'{file[:-10]}_anotated.csv'), index=False) 
                except OSError:
                    os.mkdir(os.path.join(output_path)) 
                    volume_check_csv.to_csv(os.path.join(output_path,f'{file[:-10]}_anotated.csv'), index=False) 
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