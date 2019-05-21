import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people, load_digits
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


pca = PCA(n_components=100, svd_solver='randomized', whiten=True).fit(faces.data)
pca2 = PCA(n_components=10, svd_solver='randomized', whiten=True).fit(faces.data)
pca3 = PCA(n_components=200, svd_solver='randomized', whiten=True).fit(faces.data)
# pca = RandomizedPCA(150)
# pca.fit(faces.data)

components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

components2 = pca2.transform(faces.data)
projected2 = pca2.inverse_transform(components2)

components3 = pca3.transform(faces.data)
projected3 = pca3.inverse_transform(components3)
# fig, axes = plt.subplots(4, 8, figsize=(9, 4),
#                          subplot_kw={'xticks':[], 'yticks':[]},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))

# for i, ax in enumerate(axes.flat):
#     ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')

fig, ax = plt.subplots(4, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected2[i].reshape(62, 47), cmap='binary_r')
    ax[2, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
    ax[3, i].imshow(projected3[i].reshape(62, 47), cmap='binary_r')

ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('10-dim\nreconstruction');
ax[2, 0].set_ylabel('100-dim\nreconstruction');
ax[3, 0].set_ylabel('200-dim\nreconstruction');

plt.show()

# digits = load_digits()
# digits.data.shape
# pca = PCA(2)  # project from 64 to 2 dimensions
# projected = pca.fit_transform(digits.data)
# print(digits.data.shape)
# print(projected.shape)

# plt.scatter(projected[:, 0], projected[:, 1],
#             c=digits.target, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('Pastel1', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# # plt.show()