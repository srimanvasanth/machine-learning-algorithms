from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x, y = make_regression(n_samples=100, n_features=4, n_informative=2, random_state=0, shuffle=False)

x_tr, x_te, y_tr, y_te = train_test_split(x, y, random_state=0, test_size=0.15)

reg = RandomForestRegressor(max_depth=2, random_state=0)
reg = reg.fit(x, y)

r2 = r2_score(y_te, reg.predict(x_te))
print(r2)

X, Y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, random_state=0, test_size=0.15)

clf = RandomForestClassifier(max_depth=2, random_state=0, criterion="entropy", min_samples_split=2)
clf = clf.fit(X, Y)

acc = accuracy_score(Y_te, clf.predict(X_te))
print(acc)
