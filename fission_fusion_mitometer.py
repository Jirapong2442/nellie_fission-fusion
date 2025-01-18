import numpy as np

def check_Fis_Fus_Mitometer (file_path,timeDim):
    #check fission fusion from mitometer output
    file_path = file_path
    c = np.zeros((timeDim,))
    event_per_frame = np.zeros((timeDim,))
    frame_length = timeDim
    i = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.replace('NaN', '0')
        arr = np.fromstring(line, dtype=int, sep=', ')

        if i == 0:
            c = np.stack((arr, c),axis=0)
        else: 
            c = np.vstack((c,arr))
        i += 1 
    #column =frame row =label
    # value = track number from which mitochondria fission 
    np_array = np.array(c)
    for frame in range(frame_length):
        event_per_frame[frame] = np.count_nonzero(np_array[0:,frame])

    return event_per_frame

