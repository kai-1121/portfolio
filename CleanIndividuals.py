import os
import csv
import numpy as np
from sklearn.cluster import KMeans
import glob
# file_path = "/home/user/Users/kai/data/pose_results/pose_results_1minite/track_id_20221129/"
# file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/ri_shouki/1000-1010/"
file_path = "/home/user/Users/kai/data/pose_results/watari/"

result_path = file_path + "cleaned/"
# kai chamge
os.makedirs(result_path, exist_ok=True)
num_of_points = 17
n_individual = 1
def openTrackingFile(file, n=17)-> list:
    '''	
    :param file: csv file	
    :param n: num of joints	
    :return:	
    '''
    data = []
    with open(file, newline='\r\n') as f:
        a = list(csv.reader(f, delimiter=','))
        n_results = len(a) // n
        for i in range(n_results):
            idx_start = ((n+1) * i)+1
            idx_stop = (n+1) * (i + 1)
            data.append(np.array(a[idx_start:idx_stop]).astype(float))
    return data
def cleanDuplicateData(data: list, n_individual=2, conf_th = 0.4):
    '''	
    :param data: list of data	
    :param n_individual: maximum number of individual	
    :param conf_th: the minimum confidence to be accepted	
    :return:	
    '''
    #remove all data with confidence lower than conf_th
    clean_data = []
    for d in data:
        centro = np.median(d[[0, 5, 6]], axis=0)  # centro is the average of left and right shoulder
        conf_avg = centro[-1]
        if (conf_avg > conf_th) | (len(data) == n_individual):
            clean_data.append(np.vstack([d, centro]))
    clean_data = np.array(clean_data)
    # if number of clean data less than the expected individuals remove data of that frame
    if len(clean_data) < n_individual:
        #kai start
        if len(data)==0:
            return np.zeros((n_individual, num_of_points+1, 3))#Processing when there is no data in a file
        #kai close
        return np.zeros((n_individual, len(data[0])+1, 3))
    # aggregate data based on clustering results
    kmeans = KMeans(n_clusters=n_individual, random_state=0).fit(clean_data[:, -1, 0:2])
    labels = kmeans.labels_
    n_labels=np.unique(kmeans.labels_)
    final_data = []
    for l in n_labels:
        if len(clean_data[labels == l]) == 1:
            x=clean_data[labels == l]
            y=np.squeeze(clean_data[labels == l])
            final_data.append(np.squeeze(clean_data[labels == l]))
        else:
            final_data.append(np.average(clean_data[labels == l], 0))
    return np.array(final_data, dtype=float)
all_frames_data = []
list_of_files = sorted( filter( os.path.isfile,
                        glob.glob(file_path + "*.csv") ) )
for f in list_of_files:
    print(f)
    track_results = openTrackingFile(f, num_of_points)
    clean_results = cleanDuplicateData(track_results, n_individual=n_individual)
    all_frames_data.append(clean_results)
#last row is centroid
np.save(result_path + "cleaned_tracking.npy", np.array(all_frames_data, dtype=float))