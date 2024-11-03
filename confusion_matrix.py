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

folder = "diff_15_comb_30"
dir_path = "/home/jirapong/nellie/my_script/gridsearch/"
path = os.path.join(dir_path,folder)
output_all = []

for file in os.listdir(path):
    df = pd.read_csv(os.path.join(path,file))
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









