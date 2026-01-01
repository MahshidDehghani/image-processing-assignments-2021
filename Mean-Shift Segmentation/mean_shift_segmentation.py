import cv2
import numpy as np


def mean_shift_segmentation(image, spatial_radius=10, color_radius=20, max_iter=10):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w, c = img.shape

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    features = np.zeros((h, w, 5), dtype=np.float32)
    features[:, :, 0] = X
    features[:, :, 1] = Y
    features[:, :, 2:] = img

    # Flatten features
    flat_features = features.reshape(-1, 5)
    shifted_features = np.copy(flat_features)

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")
        for i, f in enumerate(shifted_features):
            # Compute distance in spatial domain
            spatial_diff = flat_features[:, :2] - f[:2]
            spatial_dist = np.sum(spatial_diff ** 2, axis=1)

            # Compute distance in color domain
            color_diff = flat_features[:, 2:] - f[2:]
            color_dist = np.sum(color_diff ** 2, axis=1)

            # Apply kernel (within radius)
            mask = (spatial_dist < spatial_radius ** 2) & (color_dist < color_radius ** 2)
            neighbors = flat_features[mask]

            if len(neighbors) > 0:
                shifted_features[i] = np.mean(neighbors, axis=0)

    shifted_img = shifted_features[:, 2:].reshape(h, w, 3).astype(np.uint8)
    segmented_image = cv2.cvtColor(shifted_img, cv2.COLOR_LAB2BGR)

    return segmented_image


image = cv2.imread("park.jpg")
image = cv2.resize(image, (0, 0), fx=0.05, fy=0.05)
segmented = mean_shift_segmentation(image)
cv2.imwrite("res05.jpg", segmented)