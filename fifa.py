import pandas as pd
import numpy as np

df = pd.read_csv('FIFA18.csv')

df = df.fillna(value = 0)

df.head()

df = df.drop(['Date','Team','Opponent'], axis = 1)

X = df.iloc[:, :17].values
y = df.iloc[:, 17].values

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 100)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# logistic regression
# -------------------

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print('accuracy: ', accuracy_score(y_test, y_pred)*100, '%\n\n')

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

new_pred = lr.predict(sc.transform(np.array([[0, 60, 6, 0, 3, 3, 2, 1, 25, 2, 86, 511, 105, 10, 0, 0, 0]])))

comparison = pd.DataFrame({
    'actual result (man of the match)'   : y_test,
    'machine predict (man of the match)' : y_pred
})

