import numpy as np
import pandas as pd
import os

def predict(check_label,mini_df):
    '''
    function to query frame that has event. Fusion will be original frame, fission will be frame - 1
    output the frame that has event
    '''

    working_arr = mini_df[mini_df["Label"] == int(check_label)]
    working_arr = np.array(working_arr)

    #change the assignment frame to t for fission
    # event assignment for fusion will be at frame t, but those of fission will be at frame t+1 
    # so the statement below will change the frame of fission to t making both fission/fusion at the same frame (prior to the event)
    working_arr[working_arr[:, 2] == 0, 0] -= 1

    output = np.unique(working_arr[:,0])

    return output

#put event on the frame that has fission/fusion
def annotate (check_label,mini_df,file):
    '''
    function to fill in row that has fission/fusion event to corresponding frame from def predict
    '''
    df = file[file["Label"] == int(check_label)]
    df = np.array(df)
    df[:,8] = 0

    output = predict(check_label,mini_df)

    for x in output:
        df[df[:, 6] == x, 8] = 1
    return df

def run_all(event_path, annotation_path, output_path, labels):
    index = 0
    for file in os.listdir(event_path):
        if "event" in file:
            event = pd.read_csv(os.path.join(event_path,file))
            mini_df = event[["Frame","Label","isFusion"]]
            #2 output that manually annotated
            output1 = predict(labels[index],mini_df)
            output2 = predict(labels[index+1],mini_df)
            
        if file[:-10] + "glu.xlsx" in os.listdir(annotation_path): 
            annotation = pd.read_excel(os.path.join(annotation_path,file[:-10] + "glu.xlsx"),sheet_name = "Sheet1") 
            column = annotation.columns
            df_1 = annotate(labels[index],output1, annotation)
            df_2 = annotate(labels[index+1],output2, annotation)
            
            output_annotate = np.vstack((df_1,df_2))
            volume_check_csv = pd.DataFrame(output_annotate, columns= column)
            try: 
                volume_check_csv.to_csv(os.path.join(output_path,f'{file[:-10]}_anotated.csv'), index=False) 
            except OSError:
                os.mkdir(os.path.join(output_path)) 
                volume_check_csv.to_csv(os.path.join(output_path,f'{file[:-10]}_anotated.csv'), index=False) 
            index += 2


if __name__ == "__main__":

    
    diff_threshold = ['0.0','0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0',
                    '2.25','2.5','2.75','3.0','3.25','3.5','3.75','4.0','4.25',
                    '4.5','4.75','5.0']

    comb_threshold = ['0', '5', '10', '15', '17', '19' , '21', '23','25', '31', '50', '75', '100'] #,
    # done '17','19','21','23','25','27','31'
    for i in range(len(comb_threshold)):
        comb = comb_threshold[i]
        output_folder = './gridsearch10s_n2/test_neighbor_' + comb
        #os.mkdir(output_folder) 

        for j in range(len(diff_threshold)):
            diff = diff_threshold[j]
            path = "./all_10s_n2/" + diff + "_" + comb + "/"
            annotation_path = "./template/glu10s"
            output_path =  output_folder +'/diff_' + diff + "_comb_" + comb
            #labels = [80,323,200,345,26,255, 78,296, 250,304,198,321,31,249,268,301,] # 10 0 first
            #labels = [ 78,296, 26,255, 250,304,198,321, 
            #        268,301,31,249, 80,323,200,345,] # 1 0 first
            labels = ['18','55','99','152','58','75','144','315','298','336','361','342', 
                     '35', '129','163' ,'94','29','89','133','19','65','198','241','338',
                     '25','73','178','177','199','96','304','278','249','404','416','420',
                     '110','139','215','180','236','229','371','305','448','486','506','507']
            index = 0
            
        
            for file in os.listdir(path):
                if "event" in file:
                    event = pd.read_csv(os.path.join(path,file))
                    mini_df = event[["Frame","Label","isFusion"]]
                    #2 output that manually annotated repeat is the best way since the dimension of output is unknown
                    # if we concatenate everthing together it would be difficult to separate each event

                    #need to find the way to automate this -> cannot for loop and append to one array since the dim are different
                    '''
                    output1 = predict(labels[index],mini_df)
                    output2 = predict(labels[index+1],mini_df)
                    output3 = predict(labels[index+2],mini_df)
                    output4 = predict(labels[index+3],mini_df)
                    output5 = predict(labels[index+4],mini_df)
                    output6 = predict(labels[index+5],mini_df)
                    output7 = predict(labels[index+6],mini_df)
                    output8 = predict(labels[index+7],mini_df)
                    output9 = predict(labels[index+8],mini_df)
                    output10 = predict(labels[index+9],mini_df)
                    output11 = predict(labels[index+10],mini_df)
                    output12 = predict(labels[index+11],mini_df)
                    '''
                    
                if file[:-10] + "glu.xlsx" in os.listdir(annotation_path): 
                    annotation = pd.read_excel(os.path.join(annotation_path,file[:-10] + "glu.xlsx"),sheet_name = "Sheet1") 
                    column = annotation.columns
                    df_1 = annotate(labels[index],mini_df,annotation)
                    df_2 = annotate(labels[index+1],mini_df,annotation)
                    df_3 = annotate(labels[index+2],mini_df,annotation)
                    df_4 = annotate(labels[index+3],mini_df,annotation)
                    df_5 = annotate(labels[index+4],mini_df,annotation)
                    df_6 = annotate(labels[index+5],mini_df,annotation)
                    df_7 = annotate(labels[index+6],mini_df,annotation)
                    df_8 = annotate(labels[index+7],mini_df,annotation)
                    df_9 = annotate(labels[index+8],mini_df,annotation)
                    df_10 = annotate(labels[index+9],mini_df,annotation)
                    df_11 = annotate(labels[index+10],mini_df,annotation)
                    df_12 = annotate(labels[index+11],mini_df,annotation)
                    
                    output_annotate = np.vstack((df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,df_12))
                    volume_check_csv = pd.DataFrame(output_annotate, columns= column)
                    try: 
                        volume_check_csv.to_csv(os.path.join(output_path,f'{file[:-10]}_anotated.csv'), index=False) 
                    except OSError:
                        os.makedirs(output_path)
                        volume_check_csv.to_csv(os.path.join(output_path,f'{file[:-10]}_anotated.csv'), index=False) 
                    #update index
                    index += 12
     
