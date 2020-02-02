from sklearn.datasets import load_iris
#从Scikit-Learn库导入集成学习模型的随机森林分类算法
from sklearn.ensemble import RandomForestClassifier
#载入鸢尾花数据集
X, y = load_iris(return_X_y=True)
#训练模型
#随机森林和决策树算法一样，同样其中有一个名为“criterion”的参数
#同样可以传入字符串“gini”或“entropy”，默认使用的是基尼指数
clf = RandomForestClassifier().fit(X, y)
#使用模型进行分类预测
print(clf.predict(X))
#性能得分
print(clf.score(X,y))