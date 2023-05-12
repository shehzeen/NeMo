import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
import re
import torch

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# set the directory path containing the embedding files
content_embedding_path = "/data/shehzeen/SSLTTS/SupDataDirs/NewSpecPreprocessing/EnglishSpanishBengCelebEpoch39/"

# compile a regular expression pattern to match the desired file names
pattern = 'embedding_content_embedding'
# pattern = re.compile('embedding_content_embedding')

embedding_files=[]
counter = 0

for f in os.listdir(content_embedding_path):
    if (f.startswith(pattern) and counter<10000):
        embedding_files.append(str(f))
        counter +=1
        
# print(embedding_files)

# initialize an empty list to store the embeddings
embeddings = []

# loop over each file and load its contents into the embeddings list
for file in embedding_files:
    embedding = torch.load(os.path.join(content_embedding_path, file))
    embeddings.append(embedding.T)
    
concatenated_tensor = torch.cat(embeddings, dim=0)
print("Concatenated tensor shape: ", concatenated_tensor.shape)
# convert the embeddings to a numpy array
embeddings_array = concatenated_tensor.numpy()
print("Embeddings array shape: ", embeddings_array.shape)

# set number of clusters for k-means
num_clusters = 100

# initialize k-means algorithm with number of clusters and random initialization
kmeans = MiniBatchKMeans(n_clusters=num_clusters, init='random')

# fit k-means to embeddings
kmeans.fit(embeddings_array)

# get cluster labels for each embedding
cluster_labels = kmeans.labels_

# print the cluster labels
# print(cluster_labels)