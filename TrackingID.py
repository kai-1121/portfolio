import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

file_path = "/home/user/Users/kai/data/pose_results/a_school/sub_rs/1000-1010/cleaned/"
clean_data = np.load(file_path + "cleaned_tracking.npy")

missing_value_num=0
num_id = len(clean_data[0])
all_tracking_data = np.copy(clean_data)
for i in range(len(clean_data) - 1):
   t_data = all_tracking_data[i] #current data
   next_data = clean_data[i+1] #next data


#dist=[array([[id0to0,0to1]]),array([[id1to0,1to1]])] when num_id=2
   dist =  np.array([distance_matrix(np.expand_dims(d[-1, 0:2], 0), next_data[:, -1, 0:2]) for d in t_data])
   # if i==1748:
   #    print(f"next_data[:, -1, 0:2]:{np.expand_dims(t_data[-1, 0:2], 0),}")
   #    print(f"num_id{clean_data}")
   # print(i)
   # if the next frame is zeros (low con) or the closest individuals for the current frame are the same
   
   #pras_code
   # if (np.sum(np.argmin(dist, 0)) != 1) |(np.sum(next_data) == 0):
   #    all_tracking_data[i + 1] = t_data
   #    count+=1
   # else:
   #    all_tracking_data[i+1] = next_data[np.argmin(dist, 0)]
   
   #kai_code start
   if (np.sum(next_data) == 0):
      lll=np.argmin(dist, 0)
      xxx=np.sum(np.argmin(dist, 0))
      t_data[:,:,2]=-1
      all_tracking_data[i + 1] = t_data
      missing_value_num+=1
   #when num_id=2,it can be used
   # elif np.argmin(dist) == 2 or 3:#dist=[array([[id0to0,0to1]]),array([[id1to0,1to1]])] when num_id=2
   # elif (dist[0,0,1]+dist[1,0,0]) < (dist[0,0,0]+dist[1,0,1]):
   #    all_tracking_data[i+1,0]=next_data[1]
   #    all_tracking_data[i+1,1]=next_data[0]
   else:
      pass
   #kai_code end
print(f"missing_value_num:{missing_value_num}:frame_num{i}")

#save as dataframe

df = pd.DataFrame()
for i in range(num_id):
   df[str(i)] = list(all_tracking_data[:, i, :, :])
   # df[str(i)] = list(clean_data[:, i, :, :])

df.to_pickle(file_path + "tracking_results.pkl")


#checkk sanity
import matplotlib.pyplot as plt

figure, axis = plt.subplots(2)

# axis[0].plot(all_tracking_data[:, 1, -1, 0])
# axis[1].plot(all_tracking_data[:, 1, -1, 1])
# axis[0].plot(all_tracking_data[:, 0, 3, 0])
# axis[1].plot(all_tracking_data[:, 0, 3, 1])

axis[0].plot(clean_data[:, 0, 4, 0], label='f(n)', color='red')
# axis[0].plot(clean_data[:, 1, 4, 0], label='f(n)', color='green')
axis[0].plot(all_tracking_data[:, 0, 4, 0], label='f(n)', color='blue')
axis[0].set_ylabel("X coordinate [pixel]", fontsize=24)
axis[0].tick_params(labelsize=18)
axis[0].grid()

axis[1].plot(clean_data[:, 0, 4, 1], label='f(n)', color='red')
# axis[1].plot(clean_data[:, 1, 4, 1], label='f(n)', color='green')
axis[1].plot(all_tracking_data[:, 0, 4, 1], label='f(n)', color='blue')
axis[1].set_xlabel("Frame", fontsize=24)
axis[1].set_ylabel("Y coordinate [pixel]", fontsize=24)
axis[1].tick_params(labelsize=18)
axis[1].grid()

plt.show()

# save_path=file_path
# plt.savefig(save_path + '\\' + '_Transition_' + '.png', dpi=300)


