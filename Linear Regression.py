import numpy
import random
import matplotlib.pyplot as plt
import seaborn

numpy.random.seed(42)
ages = []

for i in range(250):
    ages.append(random.randint(18, 55))
nw = [6.25*i + numpy.random.normal(scale=30) for i in ages]

ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
nw = numpy.reshape(numpy.array(nw), (len(nw), 1))

def mod(ages_tr, nw_tr):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(ages_tr, nw_tr)
    return reg

from sklearn.model_selection import train_test_split
ages_tr, ages_te, nw_tr, nw_te = train_test_split(ages, nw)
reg1 = mod(ages_tr, nw_tr)
print("coefficient: ", reg1.coef_)
print("intercept: ", reg1.intercept_)
print(reg1.score(ages_tr, nw_tr))
print(reg1.score(ages_te, nw_te))

plt.figure(figsize=(10, 10))
plt.scatter(ages_tr, nw_tr, color="r", label="Train data")
seaborn.regplot(x = ages_te, y = nw_te, scatter=True, color="b", marker="s")
plt.xlabel("Ages")
plt.ylabel("Net worth")
plt.legend(loc=2)
plt.plot(ages_te,reg1.predict(ages_te))
plt.title("Graph Data")
plt.show()




