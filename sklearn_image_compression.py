# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:43:44 2025

@author: laket
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image = plt.imread('colorful.jpg')
image_array = np.array(image)

# Reshape the image array to a 2D array of pixels
width, height, channels = image_array.shape
print(width, height, channels)
pixels = image_array.reshape(-1, channels).astype(np.float32)  # Convert to float32 for better precision

# Define the number of clusters (colors) for compression
n_colors = 30

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
kmeans.fit(pixels)

# Get the cluster centers (representative colors) and scale back to [0, 255]
cluster_centers = kmeans.cluster_centers_.astype(np.float32)
cluster_centers = np.clip(cluster_centers, 0, 255).astype(np.uint8)  # Clip values to [0, 255] and convert to uint8

# Assign each pixel to its nearest cluster center
labels = kmeans.predict(pixels)

# Replace each pixel with its cluster center
compressed_pixels = cluster_centers[labels].reshape(width, height, channels)

# Create a new image from the compressed pixel data
compressed_image = Image.fromarray(compressed_pixels.astype(np.uint8))

# Display original and compressed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title('Compressed Image')
plt.imshow(compressed_image)

plt.show()

# Save the compressed image
compressed_image_path = 'comp_colorful_image.jpg'
compressed_image.save(compressed_image_path)

print(f"Original image size: {len(image_array.flatten())} bytes")
print(f"Compressed image size: {len(compressed_pixels.flatten())} bytes")
