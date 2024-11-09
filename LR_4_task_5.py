import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# Завантаження даних
m = 100
x = 6 * np.random.rand(m, 1) - 5
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(x)

print("X[0] =", x[0])
print("x_poly =", X_poly)

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_poly, y)
intercept = lin_reg.intercept_,
coef = lin_reg.coef_
print("Intercept (θ0):", intercept)
print("Коефіцієнти (θ1, θ2):", coef)

# Прогнозування на основі поліноміальних ознак
x_new = np.linspace(min(x), max(x), 100).reshape(100, 1)
x_new_poly = poly_features.transform(x_new)
y_new = lin_reg.predict(x_new_poly)

# Побудова графіка
plt.scatter(x, y, color='blue', label='Дані')
plt.plot(x_new, y_new, color='red', label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Поліноміальна регресія (степінь 2)')
plt.legend()
plt.show()

