import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 定义指数函数形式
def exponential_function(x, a, b):
    return a * np.exp(-0.1 * x) + b

# 提供两个点的坐标
x_data = np.array([0, 80])
y_data = np.array([1, 0])

# 使用 curve_fit 函数拟合数据以获得参数
params, covariance = curve_fit(exponential_function, x_data, y_data, maxfev=5000)

# 提取参数值
a, b = params

# 打印参数值
print(f"a: {a}, b: {b}")

# 绘制拟合曲线
x_fit = np.linspace(0, 30, 100)
y_fit = exponential_function(x_fit, a, b)

plt.scatter(x_data, y_data, label='Data Points')
plt.plot(x_fit, y_fit, label='Fitted Exponential Curve', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
