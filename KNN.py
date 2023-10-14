from sklearn.neighbors import KNeighborsClassifier

x = [[0],[10],[20],[30]]
y = [0,0,1,1]

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(x, y)
print(clf.predict([[5],[15],[25],[45]]))
