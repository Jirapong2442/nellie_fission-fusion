import numpy as np  
import pandas as pd
import os
from checkFissFus2 import get_fiss_fus

folder = "./self_event/"

files = os.listdir(folder)
output_fission = []
output_fusion = []
fission_df = []
fusion_df = []
fiss_fus_ratios = []
for file in files:
    if "fission" in file:
        output_fission.append(folder  + file)
    if "fusion" in file:
        output_fusion.append(folder  + file)

fission_df = get_fiss_fus(output_fission,fission_df)
fusion_df = get_fiss_fus(output_fusion,fusion_df)

for i in range(len(fusion_df)):
    for j in range(len(fusion_df[i])):
        try:
            fiss_fus_ratios.append(fission_df[i][j] / fusion_df[i][j])
        except ZeroDivisionError:
            fiss_fus_ratios.append(0)


