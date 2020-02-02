from sklearn.datasets import load_iris
#从Scikit-Learn库导入朴素贝叶斯模型中的多项式朴素贝叶斯分类算法
from sklearn.svm import SVC
#载入鸢尾花数据集
X, y = load_iris(return_X_y=True)
#训练模型
clf = SVC().fit(X, y)
#默认为径向基rbf，可通过kernel查看
print(clf.kernel)
#使用模型进行分类预测
print(clf.predict(X))
#性能得分
print(clf.score(X,y))