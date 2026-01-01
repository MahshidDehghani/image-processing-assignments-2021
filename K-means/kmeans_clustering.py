import sys
import matplotlib.pyplot as plt
import numpy as np


def cartesian_to_polar(point):
    x, y = point
    theta = np.arctan(x/y)
    rho = euclidean_distance(point, (0, 0))
    return theta, rho


def convert_points_to_polar(n, points):
    polar_points = np.zeros((n, 2))
    for i in range(n):
        point = points[i, :]
        polar_point = cartesian_to_polar(point)
        polar_points[i, :] = polar_point
    return polar_points


def load_points_from_file(file_path):
    file = open(file_path, 'r')
    k_str = file.readline()
    n = int(k_str.strip())
    points = np.zeros((n, 2))
    i = 0
    while True:
        line = file.readline()
        if not line:
            break
        x_str, y_str = line.split(" ")
        x, y = float(x_str.strip()), float(y_str.strip())
        points[i, :] = [x, y]
        i += 1
    return n, points


def plot_points(plot_path, points, x_label='x', y_label='y'):
    X, Y = np.asarray(points[:, 0]), np.asarray(points[:, 1])
    plt.figure(figsize=(15,15))
    plt.plot(X, Y, 'o', color='red')
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.savefig(plot_path)


def plot_points_with_label(plot_path, n, points, point_labels, k, cluster_centers, x_label='x', y_label='y'):
    plt.figure(figsize=(15, 15))
    for i in range(n):
        x, y = points[i, :]
        label = int(point_labels[i])
        color = colors[label]
        plt.plot(x, y, 'o', color=color)
    for i in range(k):
        x, y = cluster_centers[i, :]
        plt.plot(x, y, '+', color='k')
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.savefig(plot_path)


def initialize_centroids(n, points, k):
    cluster_centers = np.zeros((k, 2))
    indices = np.random.randint(n, size=k)
    i = 0
    for index in indices:
        cluster_centers[i, :] = points[index, :]
        i += 1
    return cluster_centers


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def assign_nearest_centroid(point, cluster_centers):
    label = 0
    min_distance = sys.maxsize
    for i in range(cluster_centers.shape[0]):
        center = cluster_centers[i, :]
        d = euclidean_distance(point, center)
        if d < min_distance:
            min_distance = d
            label = i
    return label


def assign_cluster_labels(points, cluster_centers):
    point_labels = np.zeros(n)
    for i in range(points.shape[0]):
        point = points[i, :]
        label = assign_nearest_centroid(point, cluster_centers)
        point_labels[i] = label
    return point_labels


def compute_centroids(n, points, k, cluster_centers):
    point_labels = assign_cluster_labels(points, cluster_centers)
    average_cluster_centers = np.zeros((k, 2))
    num_points_in_clusters = np.zeros(k)
    for i in range(n):
        x, y = points[i, :]
        label = int(point_labels[i])
        average_cluster_centers[label, 0] += x
        average_cluster_centers[label, 1] += y
        num_points_in_clusters[label] += 1
    for i in range(k):
        num = num_points_in_clusters[i]
        average_cluster_centers[i, :] = average_cluster_centers[i, :]/num
    return average_cluster_centers


def kmeans(n, points, k):
    cluster_centers = initialize_centroids(n, points, k)
    while True:
        average_cluster_centers = compute_centroids(n, points, k, cluster_centers)
        sum_center_distance = 0
        for i in range(k):
            center = cluster_centers[i, :]
            new_center = average_cluster_centers[i, :]
            sum_center_distance += euclidean_distance(center, new_center)
        if sum_center_distance/k < center_threshold:
            break
        cluster_centers = average_cluster_centers
    return cluster_centers


colors = ['b', 'g', 'r', 'c', 'm', 'y']
points_file = "Points.txt"
center_threshold = 0.1
k = 2
n, points = load_points_from_file(points_file)
plot_path = "res01.jpg"
plot_points(plot_path, points)


cluster_centers = kmeans(n, points, k)
point_labels = assign_cluster_labels(points, cluster_centers)
plot_path = "res02.jpg"
plot_points_with_label(plot_path, n, points, point_labels, k, cluster_centers)


cluster_centers = kmeans(n, points, k)
point_labels = assign_cluster_labels(points, cluster_centers)
plot_path = "res03.jpg"
plot_points_with_label(plot_path, n, points, point_labels, k, cluster_centers)


polar_points = convert_points_to_polar(n, points)
cluster_centers = kmeans(n, polar_points, k)
point_labels = assign_cluster_labels(polar_points, cluster_centers)
plot_path = "res04_polar_coordinate.jpg"
plot_points_with_label(plot_path, n, polar_points, point_labels, k, cluster_centers, x_label='theta', y_label='rho')
plot_path = "res04.jpg"
plot_points_with_label(plot_path, n, points, point_labels, k, cluster_centers)
