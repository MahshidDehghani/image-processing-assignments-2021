import sys
import cv2
import numpy as np


def x_gradient(gray_image):
    kernel = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return cv2.filter2D(gray_image, cv2.CV_8U, kernel)


def y_gradient(gray_image):
    kernel = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return cv2.filter2D(gray_image, cv2.CV_8U, kernel)


def gradient_magnitude(gray_image):
    x = x_gradient(gray_image)
    y = y_gradient(gray_image)
    magnitude = np.power(x, 2)+np.power(y, 2)
    return magnitude


def create_feature_vectors(image):
    im_h, im_w = image.shape[:2]
    x = np.arange(im_h)
    y = np.arange(im_w)
    yy, xx = np.meshgrid(y, x)
    feature_vectors = np.zeros((im_h, im_w, 5))
    feature_vectors[:, :, 0:3] = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    feature_vectors[:, :, 3] = xx
    feature_vectors[:, :, 4] = yy
    return feature_vectors


def compute_step_size(im_h, im_w, k):
    step = int(np.sqrt(im_h * im_w / k))
    return step


def find_local_min_gradient(gradient_image, center_x, center_y, r):
    im_h, im_w = gradient_image.shape[:2]
    min_gradient = sys.maxsize
    min_x, min_y = center_x, center_y
    for x in range(center_x - r, center_x + r + 1):
        for y in range(center_y - r, center_y + r + 1):
            if 0 <= x < im_h and 0 <= y < im_w:
                if gradient_image[x, y] < min_gradient:
                    min_gradient = gradient_image[x, y]
                    min_x, min_y = x, y
    return min_x, min_y


def initialize_cluster_centers(gray_image, k, r):
    im_h, im_w = gray_image.shape[:2]
    gradient_image = gradient_magnitude(gray_image)
    center_mask = np.zeros((im_h, im_w))
    step = compute_step_size(im_h, im_w, k)
    centers = []
    for x in range(step, im_h, step):
        for y in range(step, im_w, step):
            center_x, center_y = find_local_min_gradient(gradient_image, x, y, r)
            center_mask[center_x, center_y] = 1
            centers.append([center_x, center_y])
    return centers, center_mask, step


def draw_cluster_centers(image, centers):
    copy_image = np.copy(image)
    for center in centers:
        center_x, center_y = center
        cv2.circle(copy_image, (center_y, center_x), 5, (0,0,255), -1)
    return copy_image


def color_distance(feature_vector1, feature_vector2):
    d_l = feature_vector1[0] - feature_vector2[0]
    d_a = feature_vector1[1] - feature_vector2[1]
    d_b = feature_vector1[2] - feature_vector2[2]
    d_lab = d_l**2 + d_a**2 + d_b**2
    return d_lab


def spatial_distance(feature_vector1, feature_vector2):
    d_x = feature_vector1[3] - feature_vector2[3]
    d_y = feature_vector1[4] - feature_vector2[4]
    d_xy = d_x**2 + d_y**2
    return d_xy


def total_distance(alpha, feature_vector1, feature_vector2):
    d_lab = color_distance(feature_vector1, feature_vector2)
    d_xy = spatial_distance(feature_vector1, feature_vector2)
    return d_lab + alpha * d_xy


def distance_from_center(alpha, feature_vectors, center_feature_vector):
    d_l = feature_vectors[:, :, 0] - center_feature_vector[0]
    d_a = feature_vectors[:, :, 1] - center_feature_vector[1]
    d_b = feature_vectors[:, :, 2] - center_feature_vector[2]
    d_lab = d_l ** 2 + d_a ** 2 + d_b ** 2
    d_x = feature_vectors[:, :, 3] - center_feature_vector[3]
    d_y = feature_vectors[:, :, 4] - center_feature_vector[4]
    d_xy = d_x ** 2 + d_y ** 2
    return d_lab + alpha * d_xy


def get_neighbor_interval(im_h, im_w , center, step, r):
    center_x, center_y = center
    x_start, x_end, y_start, y_end = center_x - step - r, center_x + step + r, center_y - step - r, center_y + step + r
    if x_start < 0:
        x_start = 0
    if x_end > im_h:
        x_end = im_h
    if y_start < 0:
        y_start = 0
    if y_end > im_w:
        y_end = im_w
    return x_start, x_end, y_start, y_end


def assign_pixels_to_clusters(feature_vectors, centers, step, alpha, r):
    im_h, im_w = feature_vectors.shape[:2]
    distances = sys.maxsize * np.ones((im_h, im_w))
    labels = np.zeros((im_h, im_w))
    l = 0
    for center in centers:
        center_x, center_y = center
        x_start, x_end, y_start, y_end = get_neighbor_interval(im_h, im_w , center, step, r)
        last_distances = distances[x_start:x_end, y_start:y_end]
        last_labels = labels[x_start:x_end, y_start:y_end]
        new_distances = distance_from_center(alpha, feature_vectors[x_start:x_end, y_start:y_end, :], feature_vectors[center_x, center_y, :])
        index_array = np.greater(last_distances, new_distances)
        last_distances[index_array] = new_distances[index_array]
        last_labels[index_array] = l
        l += 1
    return labels


def draw_segmentation_boundaries(image, labels):
    im_h, im_w = image.shape[:2]
    slic = np.copy(image)
    # labels = cv2.medianBlur(labels,5)
    horizontal_edges = np.zeros((im_h, im_w))
    horizontal_edges[0:im_h-1, :] = labels[0:im_h - 1, :] - labels[1:im_h, :]
    slic[horizontal_edges != 0, :] = [0, 0, 255]
    vertical_edges = np.zeros((im_h, im_w))
    vertical_edges[:, 0:im_w-1] = labels[:, 0:im_w-1] - labels[:, 1:im_w]
    slic[vertical_edges != 0, :] = [0, 0, 255]
    return slic


img = cv2.imread("slic.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r = 5

alpha = 0.1
k = 64
centers, center_mask, step = initialize_cluster_centers(gray, k, r)
print(len(centers))
feature_vectors = create_feature_vectors(img)
labels = assign_pixels_to_clusters(feature_vectors, centers, step, alpha, r)
cv2.imwrite("res06.jpg", cv2.normalize(draw_segmentation_boundaries(img, labels), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))


alpha = 0.1
k = 256
centers, center_mask, step = initialize_cluster_centers(gray, k, r)
print(len(centers))
feature_vectors = create_feature_vectors(img)
labels = assign_pixels_to_clusters(feature_vectors, centers, step, alpha, r)
cv2.imwrite("res07.jpg", cv2.normalize(draw_segmentation_boundaries(img, labels), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))


alpha = 0.2
k = 1024
centers, center_mask, step = initialize_cluster_centers(gray, k, r)
print(len(centers))
feature_vectors = create_feature_vectors(img)
labels = assign_pixels_to_clusters(feature_vectors, centers, step, alpha, r)
cv2.imwrite("res08.jpg", cv2.normalize(draw_segmentation_boundaries(img, labels), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))


alpha = 0.3
k = 2048
centers, center_mask, step = initialize_cluster_centers(gray, k, r)
print(len(centers))
feature_vectors = create_feature_vectors(img)
labels = assign_pixels_to_clusters(feature_vectors, centers, step, alpha, r)
cv2.imwrite("res09.jpg", cv2.normalize(draw_segmentation_boundaries(img, labels), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
