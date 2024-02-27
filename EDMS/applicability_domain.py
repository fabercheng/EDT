from sklearn.neighbors import NearestNeighbors
import numpy as np
from edfp import edfp_generator

edfp_outputs = edfp_generator

k = 13
knn = NearestNeighbors(n_neighbors=k)

knn.fit(edfp_outputs)

distances, _ = knn.kneighbors(edfp_outputs)
average_distances = np.mean(distances, axis=1)

threshold = np.percentile(average_distances, 95)

is_in_domain = average_distances <= threshold

for idx, in_domain in enumerate(is_in_domain):
    print(f"EDFP {idx}: {'In domain' if in_domain else 'Out of domain'}, Average Distance: {average_distances[idx]}")
