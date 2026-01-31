import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import  LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix

irisData = pd.read_csv(r"C:\Users\Dixie\Documents\NTUST\Special Topic II\KNN\Main\iris.txt",delim_whitespace=True, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','class'])

X = irisData.iloc [:,0:1].to_numpy()
y = irisData.iloc [:,4].to_numpy()
#print (X)
loo = LeaveOneOut()
loo.get_n_splits(X) #150

distance = []
label = []
pred_label = []

for train_index, test_index in loo.split(X):

    X_train, X_test = X[train_index,:], X[test_index,:] 
    y_train, y_test = y[train_index], y[test_index]
    
    #Finding distances
    point = (X_train)-(X_test)
    distances = np.sqrt(np.sum((point)**2,axis=1))
    distance.append(distances)

    #Store value of labels
    y_labels = (y_test)
    #print (y_labels)
    label.append(list(y_labels))

#Insert nearest Neighbot
k = 3
for i in range (0,(X.shape[0])):
    nearest_neighbor = np.argsort(distance,axis=0)[i][:k]
    nearest_neighbor_label = y[nearest_neighbor]
    r=Counter(nearest_neighbor_label).most_common(1)[0][0]
    pred_label.append(r)
    print(nearest_neighbor)
    print(nearest_neighbor_label)

#print (confusion_matrix(label,pred_label))
#print (accuracy_score(label,pred_label)*100)
    
    
print (nearest_neighbor)



    

    






