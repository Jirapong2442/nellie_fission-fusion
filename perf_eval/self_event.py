import numpy as np
import pandas as pd 
from postprocessing import check_mito_number
import os
stream_frame = ['1','1','1','1','5','5','10','10']
stream_con = ['0','0.5','1','2','1','0.5','0','2']
main_dir = "/home/jirapong/jirapong/archrive/nellie_output/"
out_path = "/home/jirapong/nellie/my_script/self_event"

for s in range(len(stream_con)):
    im_concen = stream_con[s]
    im_frame = stream_frame[s]
    print(im_concen+"_"+im_frame)
        
    output_name = im_frame + "_" + im_concen + "self"

    file_path_feature = main_dir + "/" + im_frame + "_resized_" + im_concen + "_glu_1min_#1.ome-ch0-features_components.csv"
    #file_path_feature = main_dir + "1_resized_0.5_glu_1min_#1.ome-ch0-features_components.csv"
    nellie_df = pd.read_csv(file_path_feature)

    first_frame_label =  len(nellie_df['reassigned_label_raw'].unique())
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

