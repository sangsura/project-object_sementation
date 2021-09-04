import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = mpimg.imread('gaixinh.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show() 
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
print(img.shape[2])
kmeans=KMeans(n_clusters=3,random_state=0).fit(X)

label = kmeans.predict(X)
img4 = np.zeros_like(X)
# replace each pixel by its center
for k in range(3):
    img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
img4[label== (0 or 1)]=[0,0,0]
    
img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
plt.imshow(img5, interpolation='nearest')
plt.axis('off')
plt.show()
