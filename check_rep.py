import numpy as np
import pandas as pd
import os

path= "./template/"

for file in os.listdir(path):
    annotation = pd.read_excel(os.path.join(path,file)) 
    mini_df = annotation[["Label", "Frame" , "ground truth"]]
    mini_df = np.array(mini_df)
    print(np.unique(mini_df,axis = 0))
