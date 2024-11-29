
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Dataset is saved in CSV file
dataD = pd.read_csv('diabetes.csv')

# Display the top 3 samples
dataD.head(3)

dataD.describe()

dataD.shape

dataD.dtypes

# Identify NaN values
print(dataD.isna().sum())

dataD.dropna()

dataD.shape

#get features and store in X 
#get targets and store in y
X = dataD.iloc[:, :-1].values
y = dataD.iloc[:, -1].values

#split the train and test data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca)

# Scale the PCA-transformed data 
sc = StandardScaler() 
X_train_pca = sc.fit_transform(X_train_pca) 
X_test_pca = sc.transform(X_test_pca)

#training the logstic regressions modles 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifierlog = LogisticRegression(random_state = 0)
classifierlog.fit(X_train_pca, y_train)

classifierknn = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
classifierknn.fit(X_train_pca, y_train)

classifiersvc = SVC(kernel = 'linear', random_state = 0)
classifiersvc.fit(X_train_pca, y_train)


classifiersv2 = SVC(kernel = 'rbf', random_state = 0)
classifiersv2.fit(X_train_pca, y_train)

classifiernb = GaussianNB()
classifiernb.fit(X_train, y_train)

classifierdt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierdt.fit(X_train, y_train)

classifierrfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierrfc.fit(X_train, y_train)

## Building Ann
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test_pca), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 3, stop = X_set[:, 0].max() + 3, step = 1),
                     np.arange(start = X_set[:, 1].min() - 3, stop = X_set[:, 1].max() + 3, step = 1))
plt.contourf(X1, X2, classifierknn.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Features 1')
plt.ylabel('Features 2')
plt.legend()
plt.show()

# Verify pydot installation try: import pydot print("pydot is installed.") except ImportError: print("pydot is not installed. Please run `pip install pydot`.")


# Verify pydot installation
try:
    import pydot
    print("pydot is installed.")
except ImportError:
    print("pydot is not installed. Please run `pip install pydot`.")

# Verify graphviz installation
try:
    import graphviz
    print("graphviz is installed.")
except ImportError:
    print("graphviz is not installed. Please ensure it is installed and added to your PATH.")


