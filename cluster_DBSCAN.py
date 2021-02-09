import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
X = []
Y = []
for i in range(1,260):
    X.append(np.load(f'output/X{i}.npy'))
    Y.append(np.load(f'output/Y{i}.npy'))

X = np.concatenate(X,axis=0)
Y = np.concatenate(Y,axis=0)

print("X size is:",X.shape)
X = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

# Compute DBSCAN
db = DBSCAN(eps=8.5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_negative_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_negative_)

# plot
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
fig = plt.figure()
ax = Axes3D(fig)
for k, col in zip(unique_labels, colors):
    if k == -1:
        continue
    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2])


plt.show()
plt.savefig("cluster_DBSCAN.png")

