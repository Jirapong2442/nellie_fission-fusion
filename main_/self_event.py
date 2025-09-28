'''
This is an additional script (component-based algorithm) to query the intra-label fission and fusion event of each mitochondrion in the Nellie's output.

Input:
- A CSV file containing feature information (output from Nellie)
Output:
- Two CSV files containing fission and fusion events for each mitochondrion
'''


import numpy as np
import pandas as pd 
from postprocessing import check_mito_number
import os
out_path = "./self_event/"
output_name = "simulation"
main = "D:/Internship/NTU/simulation/hi_fission/"
path = main + "hi_fiss.ome-TYX-T1p0_Y0p25_X0p25-ch0-t0_to_300-features_organelles.csv"
nellie_df = pd.read_csv(path)

first_frame_label =  len(nellie_df[nellie_df['t'] == 0]['reassigned_label_raw'].unique())
max_frame_num = int(nellie_df['t'].max()) + 1

final_fission_self = np.zeros((first_frame_label,max_frame_num))
final_fusion_self = np.zeros((first_frame_label,max_frame_num))

for labels in range(1,first_frame_label):
    final_fission_self,final_fusion_self = check_mito_number(nellie_df, labels ,final_fission_self, final_fusion_self)

final_fission_self = pd.DataFrame(final_fission_self)
final_fusion_self = pd.DataFrame(final_fusion_self)
final_fission_self.to_csv(os.path.join(out_path,f'{output_name}_fission.csv'), index=False)
final_fusion_self.to_csv(os.path.join(out_path,f'{output_name}_fusion.csv'), index=False) 

print(final_fission_self,final_fusion_self)

