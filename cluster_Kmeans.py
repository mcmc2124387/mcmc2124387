import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170

# Load dataset
X = []
Y = []
for i in range(1,260):
    X.append(np.load(f'output/X{i}.npy'))
    Y.append(np.load(f'output/Y{i}.npy'))

X = np.concatenate(X,axis=0)
Y = np.concatenate(Y,axis=0)

# PCA
pca = PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

# Unevenly sized blobs
kmeans = KMeans(n_clusters=2,random_state=random_state)
labels = kmeans.fit_predict(X)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_negative_ = list(labels).count(-1)
centers = kmeans.cluster_centers_

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_negative_)

# plot
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
fig = plt.figure()
ax = Axes3D(fig)

p_center = 0
class_member_mask = (labels == 0)
max_p_number = sum(Y[class_member_mask] == 1)
for k, col in zip(unique_labels, colors):
    if k == -1:
        continue
    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    p_number = sum(Y[class_member_mask] == 1)
    print(f"Cluster{k}'s positive example number is: {p_number}")

    if p_number > max_p_number:
        p_center = k
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2])

center_labels = np.zeros(len(unique_labels),dtype=int)
center_labels[p_center] = 1

print('Cluster centers are:\n',centers)
print('Center labels are:',center_labels)
np.save('output/centers.npy',centers)
np.save('output/center_labels.npy',center_labels)

plt.title("Unevenly Sized Blobs")
plt.show()
plt.savefig("cluster_Kmeans.png")

