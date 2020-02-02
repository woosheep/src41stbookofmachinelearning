from sklearn.datasets import load_iris
#从Scikit-Learn库导入朴素贝叶斯模型中的多项式朴素贝叶斯分类算法
from sklearn.naive_bayes import MultinomialNB
#载入鸢尾花数据集
X, y = load_iris(return_X_y=True)
#训练模型
clf = MultinomialNB().fit(X, y)
#使用模型进行分类预测
print(clf.predict(X))
#性能得分
print(clf.score(X,y))