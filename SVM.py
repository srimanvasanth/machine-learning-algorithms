from sklearn.svm import SVC
import numpy as np

x = np.array([[-2,-1], [0,-1], [0,1], [2,3]])
y = np.array([0,0,1,1])

clf = SVC().fit(x, y)
print(clf.predict([[0,0], [-1,0], [3,2]]))
