import numpy as np
import pandas as pd
import os
from postprocessing import check_Fis_Fus_Mitometer

def fission_fusion_mitometer(out_dir, miter_dir):
    '''
        function to call check_Fis_Fus_Mitometer and save the output to csv file
        input:
            - out_dir: output directory
            - miter_dir: directory of mitometer result
        output:
            - None
    '''
    files = os.listdir(miter_dir)
    for file in files:
        path = os.path.join(miter_dir, file)
        event_per_frame = check_Fis_Fus_Mitometer(path)

        if 'fission' in file:
            np.savetxt(out_dir+ file[:-20] +"fission_mdivi.csv", event_per_frame, delimiter=",")
        elif 'fusion' in file:  
            np.savetxt(out_dir+ file[:-20] +"fusion_mdivi.csv", event_per_frame, delimiter=",")
    return None

if __name__ == "__main__":
    out_dir_toxin = "./mitometer_output/toxin/"
    out_dir_mdivi = "./mitometer_output/mdivi/"
    miter_dir_mdivi = "D:/Internship/NTU/data_for_script/mitometer_result/mdivi/"
    miter_dir_toxin = "D:/Internship/NTU/data_for_script/mitometer_result/toxin/"

    #fission_fusion_mitometer(out_dir, miter_dir_mdivi)
    fission_fusion_mitometer(out_dir_mdivi, miter_dir_mdivi)

    '''
    for file in files_mdivi:
        path = os.path.join(miter_dir_mdivi, file)
        event_per_frame = check_Fis_Fus_Mitometer(path)

        if 'fission' in file:
            print("Fission")
            np.savetxt(out_dir+ file[:-20] +"fission_mdivi.csv", event_per_frame, delimiter=",")
        elif 'fusion' in file:  
            print("Fusion")
            np.savetxt(out_dir+ file[:-20] +"fusion_mdivi.csv", event_per_frame, delimiter=",")

        print(event_per_frame)
        '''






