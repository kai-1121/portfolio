from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import pandas as pd

file_path_0 = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/takano_nanami/0947-0957/cleaned/"
file_path_1 = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/guenfuza_fi/0947-0957/cleaned/"

read_pickle = pd.read_pickle(file_path_0 +"Displacement_along_time.pkl")
data_0=read_pickle.values.tolist()
data_0=np.array(data_0)
data_0=np.ravel(data_0)

read_pickle = pd.read_pickle(file_path_1 +"Displacement_along_time.pkl")
data_1=read_pickle.values.tolist()
data_1=np.array(data_1)
data_1=np.ravel(data_1)

# double型の配列に変換
data_0 = data_0.astype(np.float64)
data_1 = data_1.astype(np.float64)

# s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
# s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
# path = dtw.warping_path(data_0, data_1)
# dtwvis.plot_warping(data_0, data_1, path, filename="warp.png")

# distance, paths = dtw.warping_paths(data_0, data_1)
# print(distance)
# print(paths)

d = dtw.distance_fast(data_0, data_1)
print(d)