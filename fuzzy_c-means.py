import numpy as np


def dist(x, y):
    return np.linalg.norm(x - y)


def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def make_inf_to_zero(matrix, n):
    if n == 2:
        for zero in np.argwhere(np.isinf(matrix)):
            matrix[zero[0]][zero[1]] = 0
    else:
        for zero in np.argwhere(np.isinf(matrix)):
            matrix[zero[0]] = 0

    return matrix


def normalizie_U(zeros, U):
    for zero in zeros:
        U[:, zero[1]] = 0
        U[zero[0]][zero[1]] = 1
    return U


def cal_U(points, centers, m):
    distances = ((np.abs(points - centers[:, np.newaxis]) ** 2).sum(axis=2)) ** float(1 / m - 1)
    zeros = np.argwhere(np.isinf(distances))
    distances = make_inf_to_zero(distances, 2)
    distances_sum = (1 / distances).sum(axis=0)
    distances_sum = make_inf_to_zero(distances_sum, 1)
    U = (1 / distances) / (distances_sum)
    U = normalizie_U(zeros, U)
    return U


def cal_V(points, centers, m):
    distances = ((np.abs(points - centers[:, np.newaxis]) ** 2).sum(axis=2)) ** float(1 / m - 1)
    zeros = np.argwhere(np.isinf(distances))
    distances = make_inf_to_zero(distances, 2)
    distances_sum = (1 / distances).sum(axis=0)
    distances_sum = make_inf_to_zero(distances_sum, 1)
    U = (1 / distances) / (distances_sum)
    U = normalizie_U(zeros, U)
    V = np.dot(U, points)
    return V


def fuzzy_means_c(points, m, c):
    cols_num = np.shape(points)[1]
    N = np.shape(points)[0]
    c_points = np.random.rand(c, cols_num)
    for i in range(100):
        c_points = cal_V(points, c_points, m)
    U = cal_U(points, c_points, m)
    mins = np.argmax(U, axis=0).reshape(N, 1)
    distances = ((np.abs(points - c_points[:, np.newaxis]) ** 2).sum(axis=2))
    return mins , (U * distances).sum()


def cal_cost(points, U, V):
    distances = ((np.abs(points - V[:, np.newaxis]) ** 2).sum(axis=2))
    return (U * distances).sum()

