import numpy as np
import matplotlib.pyplot as plt


def initialize_centroids(points, k):
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def make_inf_to_zero(matrix, n):
    return np.nan_to_num(matrix, neginf=0)


def normalize_U(zeros, U):
    for zero in zeros:
        U[:, zero[1]] = 0
        U[zero[0]][zero[1]] = 1
    return U


def cal_U(points, centers, m):
    distances = (np.abs(points - centers[:, np.newaxis]) ** 2).sum(axis=2)
    zeros = np.argwhere(distances == 0)
    distances = make_inf_to_zero(distances, 2)
    distances_reversed = 1 / distances
    distances_reversed_m = make_inf_to_zero(distances_reversed, 2) ** float(1 / float(m - 1))
    distances_sum = (distances_reversed_m).sum(axis=0, keepdims=1)
    distances_sum = make_inf_to_zero(distances_sum, 1)
    U = distances_reversed_m / distances_sum
    U = normalize_U(zeros, U)
    return U


def cal_V(points, centers, m):
    U = cal_U(points, centers, m)
    V = np.dot(U, points) / U.sum(axis=1, keepdims=1)
    return V


def fuzzy_means_c(points, m, c):
    c_points = initialize_centroids(points, c)
    for i in range(100):
        c_points = cal_V(points, c_points, m)
    U = cal_U(points, c_points, m)
    mins = np.argmax(U, axis=0)
    distances = ((np.abs(points - c_points[:, np.newaxis]) ** 2).sum(axis=2))
    return mins, (U * distances).sum(), c_points

