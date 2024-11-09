import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size = 0.5, random_state = 0)
regr = linear_model.LinearRegression()
regr.fit(Xtrain, Ytrain)
Ypred = regr.predict(Xtest)

print(f"Коефіцієнт регресії (regr.coef_): {regr.coef_}")
print(f"Інтерсепт (regr.intercept_): {regr.intercept_}")
print(f"R^2 (r2_score): {r2_score(Ytest, Ypred)}")
print(f"Середня абсолютна похибка (mean_absolute_error): {mean_absolute_error(Ytest, Ypred)}")
print(f"Середньоквадратична похибка (mean_squared_error): {mean_squared_error(Ytest, Ypred)}")

fig, ax = plt.subplots()
ax.scatter(Ytest, Ypred, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()