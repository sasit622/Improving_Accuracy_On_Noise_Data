import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

weights = np.zeros(X_train.shape[1])
bias = 0

learning_rate = 0.01
epochs = 100
batch_size = 16

for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        predictions = np.dot(X_batch, weights) + bias

        error = predictions - y_batch

        dw = np.dot(X_batch.T, error) / len(X_batch)
        db = np.mean(error)

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

y_pred = np.dot(X_test, weights) + bias

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Weights:", weights)
print("Bias:", bias)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Input")
plt.ylabel("Target")
plt.title("Linear Regression using Mini-Batch Gradient Descent")
plt.show()