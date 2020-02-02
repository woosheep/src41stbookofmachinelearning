#导入所需库
import matplotlib.pyplot as plt
import numpy as np

#生成数据集
x = np.linspace(-3, 3, 30)
y = 2*x + 1 
#数据集绘图
plt.scatter(x, y)
plt.show()

#从Scikit-Learn库导入线性模型中的线性回归算法
from sklearn import linear_model

#训练线性回归模型
model = linear_model.LinearRegression()
model.fit(x[:,np.newaxis], y)
#查看训练效果
print(model.coef_,model.intercept_)

#生成对比数据集
#加入模拟扰动
x_ =x+np.random.rand(30)
y = 2*x_ + 1
plt.scatter(x, y)
plt.show()

#训练线性回归模型
model = linear_model.LinearRegression()
model.fit(x[:,np.newaxis], y)
#查看训练效果
print(model.coef_,model.intercept_)