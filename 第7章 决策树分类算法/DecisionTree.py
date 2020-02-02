from sklearn.datasets import load_iris
#从Scikit-Learn库导入决策树模型中的决策树分类算法
from sklearn.tree import DecisionTreeClassifier
#载入鸢尾花数据集
X, y = load_iris(return_X_y=True)
#训练模型
clf = DecisionTreeClassifier().fit(X, y)
#使用模型进行分类预测
print(clf.predict(X))
#性能得分
print(clf.score(X,y))