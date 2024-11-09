import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import pickle

# Вхідний файл, який містить дані
input_file = 'data_multivar_regr.txt'

# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
x1, x2, x3, y = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Розбивка даних на навчальний та тестовий набори
num_training = int(0.8 * len(x1))
num_test = len(x1) - num_training

# Тренувальні дані
x1_train, x2_train, x3_train, y_train = x1[:num_training], x2[:num_training], x3[:num_training], y[:num_training]
# Тестові дані
x1_test, x2_test, x3_test, y_test = x1[num_training:], x2[num_training:], x3[num_training:], y[num_training:]

# Об'єднання ознак в матриці
X_train = np.column_stack((x1_train, x2_train, x3_train))
X_test = np.column_stack((x1_test, x2_test, x3_test))

# Створення об'єкта лінійного регресора
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

# Прогнозування результату
y_test_pred = linear_regressor.predict(X_test)

print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Збереження моделі
output_model_file = 'model3_linear.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(linear_regressor, f)

# Завантаження моделі
y_test_pred_new = linear_regressor.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

print("\nLinear regression:\n", linear_regressor.predict(np.array(datapoint)))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))
