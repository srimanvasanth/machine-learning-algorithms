from sklearn import tree

x = [[-1,-1], [0,-1], [0,1], [2,3]]
y = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
print(clf.predict([[-1,0], [0,7]]))
