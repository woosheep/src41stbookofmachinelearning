from sklearn.datasets import load_iris
#从Scikit-Learn库导入近邻模型中的KNN分类算法
from sklearn.neighbors import KNeighborsClassifier

#载入鸢尾花数据集
X, y = load_iris(return_X_y=True)
#训练模型
clf = KNeighborsClassifier().fit(X, y)
#使用模型进行分类预测
print(clf.predict(X))
#性能得分
print(clf.score(X,y))