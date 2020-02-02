#从Scikit-Learn库导入线性模型中的Logistic回归算法
from sklearn.linear_model import LogisticRegression
#Scikit-Learn库带有知名的鸢尾花分类数据集，是个分类问题的数据集
from sklearn.datasets import load_iris

#载入鸢尾花数据集
X, y = load_iris(return_X_y=True)
#训练模型
clf = LogisticRegression().fit(X, y)
#使用模型进行分类预测
print(clf.predict(X))
#性能得分
print(clf.score(X,y))