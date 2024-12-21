import numpy as np
import pandas as pd
import os

def calculate_metrics(actual, predicted):
    # True Positive (TP): Actual = 1, Predicted = 1
    TP = np.sum((actual == 1) & (predicted == 1))
    
    # True Negative (TN): Actual = 0, Predicted = 0
    TN = np.sum((actual == 0) & (predicted == 0))
    
    # False Positive (FP): Actual = 0, Predicted = 1
    FP = np.sum((actual == 0) & (predicted == 1))
    
    # False Negative (FN): Actual = 1, Predicted = 0
    FN = np.sum((actual == 1) & (predicted == 0))
    
    return TP, TN, FP, FN
diff_threshold = ['0.0','0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0',
                  '2.25','2.5','2.75','3.0','3.25','3.5','3.75','4.0','4.25',
                  '4.5','4.75','5.0']

comb_threshold = ['0', '5', '10', '15', '17', '19' , '21', '23','25', '31', '50', '75', '100']
#['17','19','21','23','25','27','31'] #,

for i in range(len(comb_threshold)):
    comb = comb_threshold[i]
    output_folder = './gridsearchn2/test_neighbor_' + comb

    for j in range(len(diff_threshold)):
        diff = diff_threshold[j]
        folder = 'diff_' + diff + "_comb_" + comb
 
        path = os.path.join(output_folder, folder)
        output_all = []

        for file in os.listdir(path):
            try:
                df = pd.read_csv(os.path.join(path,file))
            except UnicodeDecodeError:
                break
            cf = df[["Frame", "Label", "algo_output", "ground truth"]]
            cf = np.unique(cf,axis = 0)

            actual = cf[:,3]
            predict = cf[:,2]

            tp, tn, fp, fn = calculate_metrics(actual, predict)
            output = np.array([tp, tn, fp, fn])
            if len(output_all) == 0:
                output_all.append(output)
                output_all = np.array(output_all)
            else:
                output_all = np.vstack((output_all,output))

            np.save(os.path.join(path,file[:-4] + '.npy'), output) 

        np.save( os.path.join(path, folder + ".npy"),output_all)
        print(output_all)









