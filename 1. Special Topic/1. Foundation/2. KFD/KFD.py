import scipy.io
import numpy as np

import matplotlib.pyplot as plt

source = scipy.io.loadmat(r'C:\Users\Dixie\Documents\NTUST\Special Topic II\KFD\Main\data.mat')
data = source["data"]
(x,y) = data.shape #30x1500
n=y-1 #1499

total_distance = []
total_max_distance = [] 
for i in range (0,x): #30-1
    for j in range (0,n): #1499-1
        p1 = (data[i][j])
        p2 = (data[i][j+1])
        p3 = (data[i][0])
        distance = np.sqrt(((p2-p1)**2)+(j-(j+1))**2)
        total_distance.append(distance)
        max_distance = np.sqrt(((p2-p3)**2)+((j+1)-0)**2)
        total_max_distance.append(max_distance)

distance_each_row = [total_distance[i:i + 1499] for i in range (0,len(total_distance), 1499)]
max_distance_each_row = [total_max_distance[i:i + 1499] for i in range (0,len(total_max_distance), 1499)]

for i in range (0,x):
    L = np.sum(distance_each_row[i])
    d = np.amax(max_distance_each_row[i])
    a = L/n
    D = (np.log10(L/a))/(np.log10(d/a))
    print (D)
