import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# dataset
df3 = pd.read_csv("BC_dataset.csv")

# independent & dependent dataset
X = df3.drop(['id', 'Class'], axis=1).values
y = df3['Class']
X = pd.DataFrame(X)
cols = list(X.columns)
# splitting data into train & test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# logistic regression object
model = LogisticRegression()

# fitting data into model
model.fit(X_train, y_train)

file = "models/pickledModel.sav"
pickle.dump(model, open(file, 'wb'))
