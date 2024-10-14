import numpy as np
x = 1
y = 2
z = 3
a = 4
data_1 = [x,y,0]
data_2 = [z,a,-1]



for i in range(3):
    temp_list = []
    temp_list.append(data_1[i])
    temp_list.append(data_2[i])
    if i == 0:
        another_list = np.expand_dims(np.array(temp_list),axis=0)
    else:
        list_current = np.expand_dims(np.array(temp_list),axis=0)
        another_list = np.concatenate((another_list,list_current),axis =0)

print()