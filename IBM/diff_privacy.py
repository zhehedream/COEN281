from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.utils as util
from sklearn.metrics import accuracy_score
import diffprivlib.models as models
from diffprivlib.mechanisms import Laplace
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filepath = "../data/"
filename = "pseudo_facebook.csv"

names = (
    'userid',
    'age', 
    'dob_day', 
    'dob_year',
    'dob_month',
    'gender',
    'tenure',
    'friend_count',
    'friendships_initiated',
    'likes',
    'likes_received',
    'mobile_likes',
    'mobile_likes_received',
    'www_likes',
    'www_likes_received',
)
 # , sep=", ", index_col=False, engine='python'
df = pd.read_csv(filepath+filename, header=0, names=names);

# ['data': array([[],]), 'target': array([]), 'target_names': array([]), 'DESCR':""]
fb_dataset = util.Bunch()
fb_dataset.data = np.array(df)
fb_dataset.target = 


# fb_dataset.data = np.array([[df.at[row, col] for col in names] for row in df.index])
# print(fb_dataset.data)

# for row in df.index:
# 	print([df.at[row, col] for col in names])

# laplace = Laplace()
# laplace.set_epsilon(0.5)
# laplace.set_sensitivity(1)
# laplace.randomise(3)

print("----------- GaussianNB --------------")

dataset = datasets.load_iris()
print(type(dataset))
print(type(dataset.data))
# print(dataset)

# print("\n\n\n\n\n\n")
# print(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
clf = models.GaussianNB()
clf.fit(X_train, y_train)
clf.predict(X_test)
print("Test accuracy: %f" % accuracy_score(y_test, clf.predict(X_test)))

print("\n\n\n")
epsilons = np.logspace(-2, 2, 50)
bounds = [(4.3, 7.9), (2.0, 4.4), (1.1, 6.9), (0.1, 2.5)]
accuracy = list()

clf = models.GaussianNB(bounds=bounds, epsilon=0.001)
clf.fit(X_train, y_train)
print("truth:   ", y_test, " \npredict: ", clf.predict(X_test))
print("accuracy: ", accuracy_score(y_test, clf.predict(X_test)))

# for epsilon in epsilons:
#     clf = models.GaussianNB(bounds=bounds, epsilon=epsilon)
#     clf.fit(X_train, y_train)
    
#     accuracy.append(accuracy_score(y_test, clf.predict(X_test)))

# plt.semilogx(epsilons, accuracy)
# plt.title("Differentially private Naive Bayes accuracy")
# plt.xlabel("epsilon")
# plt.ylabel("Accuracy")
# plt.show()