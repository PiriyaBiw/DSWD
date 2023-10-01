
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


score_test = []
models = []

k = np.range(1, 30)
for i in k:
    classifer = KNeighborsClassifier(n_neighbors=i)
    classifer.fit(X_train, y_train)
    score_test.append()
    models.append(classifer)


pikle.dump(models, open('iris_model.pkl', 'wb'))