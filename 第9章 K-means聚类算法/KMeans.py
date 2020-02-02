#导入绘图库
import matplotlib.pyplot as plt
#从Scikit-Learn库导入聚类模型中的K-means聚类算法
from sklearn.cluster import KMeans
#导入聚类数据生成工具
from sklearn.datasets import make_blobs

#用sklearn自带的make_blobs方法生成聚类测试数据
n_samples = 1500
#该聚类数据集共1500个样本
X, y = make_blobs(n_samples=n_samples)

#进行聚类，这里n_clusters设定为3，也即聚成3个簇
y_pred=KMeans(n_clusters=3).fit_predict(X)

#用点状图显示聚类效果
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()