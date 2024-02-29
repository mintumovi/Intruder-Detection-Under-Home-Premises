# Mount the Google Drive
#from google.colab import drive
#drive.mount('/content/drive')

# Install necessary libraries
!pip install scikit-learn
!pip install matplotlib

# Import necessary libraries
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# Define the directory where the images are stored
img_dir =  "<path_to_your_repository>/Sample Dataset/"

# Read in the images from the directory
images = []
for file in os.listdir(img_dir):
  if file.endswith(".jpg"):
    img = cv2.imread(os.path.join(img_dir, file))
    images.append(img)

# Convert the images to a numpy array
images = np.array(images)

# Reshape the images to a 2D array
reshaped_images = images.reshape(images.shape[0], -1)

# Determine the optimal number of clusters using the elbow method
distortions = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=50, n_init=5, random_state=0)
  kmeans.fit(reshaped_images)
  distortions.append(kmeans.inertia_)
plt.plot(range(1, 11), distortions)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# Use KMeans clustering to separate the images into rare and frequent clusters
num_clusters = 2 # you can choose any optimal number of clusters
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=50, n_init=5, random_state=0)
cluster_labels = kmeans.fit_predict(reshaped_images)

'''
# Plot the original images with the rare frames colored in red
plt.figure(figsize=(10,10))
for i in range(images.shape[0]):
  if cluster_labels[i] == 0:
    plt.scatter(reshaped_images[i, 0], reshaped_images[i, 1], color='r')
  else:
    plt.scatter(reshaped_images[i, 0], reshaped_images[i, 1], color='b')
plt.title('Cluster-Based-Method')
plt.show()
'''

# Plot the original images with the rare frames colored in red
plt.figure(figsize=(6, 6))  # Adjust the width and height as needed
for i in range(images.shape[0]):
  if cluster_labels[i] == 0:
    plt.scatter(reshaped_images[i, 0], reshaped_images[i, 1], color='r')
  else:
    plt.scatter(reshaped_images[i, 0], reshaped_images[i, 1], color='b')
plt.title('K Means Clustering')
plt.show()

# Save the rare frames to Google Drive
rare_frames_dir = "/content/drive/MyDrive/Resultant Dataset"
if not os.path.exists(rare_frames_dir):
  os.makedirs(rare_frames_dir)
for i in range(images.shape[0]):
  if cluster_labels[i] == 0:
    cv2.imwrite(os.path.join(rare_frames_dir, "rare_frame_{}.jpg".format(i)), images[i])


#19/21
