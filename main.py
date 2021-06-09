import csv
import numpy as np
from fuzzy import fuzzy_means_c
from JSAnimation import IPython_display
from matplotlib import animation
import matplotlib.pyplot as plt


def read_data(path):
    with open(path, 'r') as f:
        data_iter = csv.reader(f,
                               delimiter=",",
                               quotechar='"')
        # data = [data for data in data_iter]
        d = []
        for row in data_iter:
            r = [float(el) for el in row]
            d.append(r)
        data_array = np.asarray(d, dtype=np.float)
        return data_array


def main():
    data = read_data("./data1.csv")
    k = []
    for c in range(1, 8):
        mins, J, c_points = fuzzy_means_c(data, 1.5, c)
        k.append(J)
    plt.plot(range(1, 8), k)
    plt.show()
    plt.scatter(data[:, 0], data[:, 1], c=[L*10+10 for L in mins], cmap='viridis')
    plt.scatter(c_points[:, 0], c_points[:, 1], c='red', cmap='viridis')
    plt.show()


main()
