from sklearn.datasets import load_iris
#从Scikit-Learn库导入神经网络模型中的神经网络分类算法
from sklearn.neural_network import MLPClassifier
#载入鸢尾花数据集
X, y = load_iris(return_X_y=True)
#训练模型
clf = MLPClassifier().fit(X, y)
#使用模型进行分类预测
print(clf.predict(X))
#性能得分
print(clf.score(X,y))