from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.utils as util
from sklearn.metrics import accuracy_score
import diffprivlib.models as models
from diffprivlib.mechanisms import Laplace
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filepath = "../Data/"
filename = "pseudo_facebook.csv"
outname = "diff_out.csv"
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

def toNum(data):
	m, n = data.shape
	for i in range(m):
		for j in range(n):
			if np.isnan(data[i][j]):
				data[i][j] = int(0)
			else:
				data[i][j] = int(data[i][j])
	return data

df = pd.read_csv(filepath+filename, header=0, names=names);

for row in df.index:
	# print(df.at[row, 'gender'])
	if df.at[row, 'gender'] == str('male'):
		# print('male')
		df.at[row, 'gender'] = int(0)
	elif df.at[row, 'gender'] == str('female'):
		# print('female')
		df.at[row, 'gender'] = int(1)
	else:
		df.at[row, 'gender'] = int(0)
		# print('other')

fb_dataset = util.Bunch()
fb_dataset.data = toNum(np.array(df))
fb_dataset.target = np.array([int(df.at[row, 'age']) for row in df.index])

print("filename", filepath+filename)
print("e-differential privacy")
X_train, X_test, y_train, y_test = fb_dataset.data, fb_dataset.data, fb_dataset.target, fb_dataset.target
epsilons = np.logspace(-2, 2, 50)
minbounds = np.amin(X_train, axis=0)
maxbounds = np.amax(X_train, axis=0)
bounds = [(minbounds[i], maxbounds[i]) for i in range(X_train[0].size)]

accuracy = list()
epsilon = 1
clf = models.GaussianNB(bounds=bounds, epsilon=epsilon)
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
# print(predict.shape)

print("epsilon: ", epsilon)
print("accuracy: ", accuracy_score(y_test, predict))
for row in df.index:
	if df.at[row, 'gender'] == 0:
		df.at[row, 'gender'] = 'male'
	elif df.at[row, 'gender'] == 1:
		df.at[row, 'gender'] = 'female'
	else:
		df.at[row, 'gender'] = 'male'

	df.at[row, 'age'] = predict[row]

print(df)
df.to_csv(filepath+outname, index=False)
