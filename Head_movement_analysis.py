import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
import datetime
import re
from pyentrp import entropy as ent
# import TimeLib

import pandas as pd
# file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/sub_GF/0947-0957/cleaned/"
file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/sub_TN/0947-0957/cleaned/"
# file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/sub_RS/0947-0957/cleaned/"
# file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/sub_MH/0947-0957/ch2/cleaned/"
# file_path = "/home/user/Users/kai/data/pose_results/20220627_5th_period/first_v2/cleaned/"
# file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/sub_MH/1010-1020/cleaned/"
# file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/sub_TN/1000-1010/cleaned/"

from scipy.interpolate import interp1d
def LinearInterpolationFunction(data):
    if len(data) == 0:
        return None
    
    completions_num=0
    for i in range(len(data)):
       if np.all(data[i,:,2]==-1):
           data[i,:,0:2]=np.nan 
           completions_num += 1
           
    print(f"completions_num:{completions_num}")
    
    time = np.arange(0, len(data[:,0,0]), 1)
    
    # 線形補完
    for i in range(17):
        x_data = pd.Series(data[:,i,0], index=time)
        y_data = pd.Series(data[:,i,1], index=time)
        data[:,i,0] = x_data.interpolate(limit_direction='both').values
        data[:,i,1] = y_data.interpolate(limit_direction='both').values
        
    return data


'''
関数：ローパスフィルターをかけて、復元した波形を返す
引数：wave（処理する信号）, fs（サンプリング周波数）, fe（カットオフ周波数）, n（フィルターの次数）
戻値：wave（処理後の信号）
Function: Low-pass filter and return the restored waveform
Arguments: wave (signal to be processed), fs (sampling frequency), fe (cutoff frequency), n (filter order)
Return: wave (signal after processing)
'''
def LowpassFilter(wave, fs, fe, n):
    wave=list(map(float, wave))
    
    nyq = fs / 2.0
    b, a = signal.butter(1, fe / nyq, btype='low')
    for i in range(0, n):
        wave = signal.filtfilt(b, a, wave)
    return np.array(wave)


def moving_average(data,window_len):
    #移動平均
    #convolveが返す結果を個数で割る方法
    # moving_average_data=np.convolve(data,np.ones(window_len), mode='valid') / window_len
    moving_average_data=np.convolve(data,np.ones(window_len) / window_len, mode='valid')
    return moving_average_data


'''
関数：パワースペクトルを求める
引数：data（信号データ）, fs（サンプリング周波数）, graph_data（グラフタイトル）, save_path（グラフの保存パス）
戻値：np.mean(np.sqrt(Pxx_den_med))（パワースペクトルの平均値）
Function: Find the power spectrum
Arguments: data (signal data), fs (sampling frequency), graph_data (graph title), save_path (graph save path)
Return: np.mean(np.sqrt(Pxx_den_med)) (mean value of power spectrum)
'''

def Movement_Trajectory(data,shoulder_point,data_x,data_y, graph_title="", save_path=""):
    #Output only right ear?
    x_keypoints=list(map(float, data_x))
    y_keypoints=list(map(float, data_y))
    
    movement_trajectory=[x_keypoints,y_keypoints]
    
    x_head=list(map(float, data[10,3:5,0:1]))/shoulder_point[0]
    y_head=list(map(float, data[10,3:5,1:2]))/shoulder_point[1]
    
    x_sholder=list(map(float, data[10,5:7,0:1]))/shoulder_point[0]
    y_sholder=list(map(float, data[10,5:7,1:2]))/shoulder_point[1]
    
    #正規化
    
    # グラフ表示
    fig = plt.figure(figsize=(12.0, 10.0))
    if len(graph_title) != 0:
        plt.title = graph_title
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # 移動軌跡
    plt.plot(x_keypoints, y_keypoints, label='f(n)', color='blue')
    plt.plot(x_head, y_head, label='f(n)', color='red')
    plt.plot(x_sholder, y_sholder, label='f(n)', color='green')
    plt.xlabel("X coordinate [pixel]", fontsize=24)
    plt.ylabel("Y coordinate [pixel]", fontsize=24)
    # plt.tick_params(width = 3.2, length = 40)
    #haba3.5,tate40
    #sub_GF
    # plt.xlim(11.1,14.5)
    # plt.ylim(100,140)
    #sub_TN
    # plt.xlim(11.6,15.1)
    # plt.ylim(-10,30)
    #sub_RS
    # plt.xlim(4.8,8.3)
    # plt.ylim(10,50)
    #sub_MH
    # plt.xlim(9,12.5)
    # plt.ylim(0,40)
    
    plt.xticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.tick_params(labelsize=18)
    plt.grid()
    
    print(f'id: {graph_title}')

    if len(save_path) != 0:
        plt.savefig(save_path + '\\' + graph_title + '_movement_trajectory' + '.png', dpi=300)
    plt.close(fig)
    time.sleep(1)

    return movement_trajectory
    

def average_speed(data_x, data_y, fps=30, graph_title="", save_path=""):
    '''
    average_speed: displacement per frame
    
    '''
    
    #Output only right ear?
    # x_keypoints=list(map(float, data[:,4,0:1]))
    # y_keypoints=list(map(float, data[:,4,1:2]))
    x_keypoints=list(map(float, data_x))
    y_keypoints=list(map(float, data_y))

    #average_speed
    x_speed_per_second=[]
    y_speed_per_second=[]
    average_speed=[]
    for i in range(len(x_keypoints)-1):
        x_speed_per_second.append(abs(x_keypoints[i+1]-x_keypoints[i]))
        y_speed_per_second.append(abs(y_keypoints[i+1]-y_keypoints[i]))
    average_speed=np.sqrt((np.array(x_speed_per_second)**2+np.array(y_speed_per_second)**2))
    Total_trajector_length=sum(average_speed)
    print(f"Total_trajector_length:{Total_trajector_length}")
    
    #正規化
    
    # グラフ表示
    fig = plt.figure(figsize=(12.0, 10.0))
    if len(graph_title) != 0:
        plt.title = graph_title
    plt.rcParams['font.size'] = 12
    
    plt.plot(x_speed_per_second, label='x_speed_per_second', color='red')
    plt.plot(y_speed_per_second, label='y_speed_per_second', color='green')
    plt.plot(average_speed, label='average_speed', color='blue')
    plt.xlabel("Frame", fontsize=24)
    plt.ylabel("speed [pixel/frame]", fontsize=24)
    plt.xticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.tick_params(labelsize=18)
    plt.grid()
    
    print(f'id: {graph_title}')
    
    plt.show()

    if len(save_path) != 0:
        plt.savefig(save_path + '\\' + graph_title + '_average_speed_' + '.png', dpi=300)
    plt.close(fig)
    time.sleep(1)

    return average_speed

def sample_entropy(data):
    '''
    data : displacement per frame
    '''
    data=data.tolist()
    #sample_entropy
    # std_data = np.std(data)
    # sample_entropy = ent.sample_entropy(data, 4, 0.2 * std_data)
    # print(f"sample_entropy:{sample_entropy[1]}")
    
    from nolds import sampen
    sample_entropy = sampen(data, 30)   
    print(f"sample_entropy:{sample_entropy}")
    
    #正規化

    return sample_entropy


def Confidence_distribution(data):
    #Output only right ear?
    conf=list(map(float, data[:,4,2]))

    #Confidence_distribution
    df = pd.DataFrame({'analysis':conf})
    desc=df.describe()
    quantile=df.quantile([0.3,0.5,0.7])
    print(desc)
    print(f"\nquantile:\n{quantile}")
    data
    return desc,quantile

def rms(data):
    '''
    data : displacement per frame
    '''
    
    return np.sqrt(sum(v*v for v in data) / len(data))


'''
関数：パワースペクトルを求める
引数：data（信号データ）, fs（サンプリング周波数）, graph_data（グラフタイトル）, save_path（グラフの保存パス）
戻値：np.mean(np.sqrt(Pxx_den_med))（パワースペクトルの平均値）
Function: Find the power spectrum
Arguments: data (signal data), fs (sampling frequency), graph_data (graph title), save_path (graph save path)
Return: np.mean(np.sqrt(Pxx_den_med)) (mean value of power spectrum)
'''

def Compute_PowerSpectrum_Welch(data, fs=30, graph_title="", save_path=""):
    data=data.tolist()
    
    # データのパラメータ
    N = len(data)    # サンプル数
    dt = 1/fs               # サンプリング間隔
    t = np.arange(0, N * dt, dt)  # 時間軸
    freq = np.linspace(0, 1.0 / dt, N)  # 周波数軸
    na_f = 1/dt/2   # ナイキスト周波数

    f = data

    # バンドパスフィルター
    #sos = signal.butter(50, [0.1, 15], 'bandpass', fs=1/dt, output='sos')
    #f = signal.sosfilt(sos, f)

    # Welch法によるパワースペクトル密度の計算（Calculation of power spectral density using the Welch method）
    f_med, Pxx_den_med = signal.welch(f, 1/dt, 'flattop', 1024, scaling='spectrum')

    # グラフ表示
    fig = plt.figure(figsize=(12.0, 10.0))
    if len(graph_title) != 0:
        plt.title = graph_title
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 時間信号（元）
    plt.subplot(211)
    # plt.plot(t, f, label='f(n)', color='blue')
    # plt.xlabel("Time", fontsize=24)
    plt.plot(f, label='f(n)', color='blue')
    plt.xlabel("Frame", fontsize=24)
    plt.ylabel("Moving Distance [pixel/frame]", fontsize=20)
    plt.xlim(-500,18000)
    plt.ylim(-0.02,0.82)
    plt.xticks(fontsize=18)
    plt.xticks(fontsize=18)
    #plt.ylim(0, 15)
    plt.tick_params(labelsize=18)
    plt.grid()

    # パワースペクトル
    plt.subplot(212)
    plt.semilogy(f_med, np.sqrt(Pxx_den_med), color='blue')
    plt.xlabel('Frequency [Hz]', fontsize=24)
    plt.ylabel('Linear spectrum [V RMS]', fontsize=20)
    plt.xlim(-0.1,16)
    plt.ylim(10**(-6),10**(-1))
    plt.xticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.grid()
    plt.tick_params(labelsize=18)

    print(f'id: {graph_title}, ps max: {np.max(np.sqrt(Pxx_den_med))}, ps mean: {np.mean(np.sqrt(Pxx_den_med))}, psd mean: {np.mean(Pxx_den_med)}')

    if len(save_path) != 0:
        plt.savefig(save_path + '\\' + graph_title + '_ps_mean_' + '.png', dpi=300)
        # plt.savefig(save_path + '\\' + graph_title + '_ps_mean_' + TimeLib.now_time() + '.png', dpi=300)
    #plt.show()
    plt.close(fig)
    time.sleep(1)

    return np.mean(np.sqrt(Pxx_den_med))


from sklearn.metrics import mean_squared_error

def analysis_still_periods(data,graph_title,save_path):
    #calculate RMSE,length of Movement_Trajectory by 5seconds
    # 時系列データ (座標)
    # 5秒ごとにデータを分割
    interval = 30 * 5 # 30fps * 5 sec
    num_intervals = len(data) // interval

    rmse_values = []
    total_movement = []
    for i in range(num_intervals):
        start = i * interval
        end = start + interval
        data_interval = data[start:end]
        rmse = np.sqrt(np.mean((data_interval - np.mean(data_interval)) ** 2))
        rmse_values.append(rmse)
        total_movement.append(np.sum(np.abs(np.diff(data_interval))))

    # rmse_values には、5秒ごとのRMSEが格納されています
    # total_movementには、5秒ごとの総移動量が格納されています

    #translate binary_waveform
    rmse_values_threshold=0.017
    total_movement_threshold=0.16
    moving_state=[]
    for i in range(num_intervals):
        if(rmse_values[i]> rmse_values_threshold and total_movement[i]>total_movement_threshold):
            moving_state.append(1)
        else:
            moving_state.append(0)
    
    binary_waveform=[]
    for i in range(num_intervals):
        x=moving_state[i]
        for j in range(interval):
            binary_waveform.append(x)
    
    #calculate still periods percentage
    # 配列の要素の値が1である個数を求める
    count = moving_state.count(1)

    # 配列に占める割合を求める
    moving_state_percentage = count / len(moving_state)
    
    # グラフ表示
    fig = plt.figure(figsize=(12.0, 10.0))
    if len(graph_title) != 0:
        plt.title = graph_title
    plt.rcParams['font.size'] = 12
    
    plt.plot(binary_waveform, label='binary_waveform', color='blue')
    plt.xlabel("Frame", fontsize=24)
    plt.ylabel("binary_waveform [pixel/frame]", fontsize=24)
    plt.xticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.tick_params(labelsize=18)
    plt.grid()
    
    print(f'id: {graph_title}')

    if len(save_path) != 0:
        plt.savefig(save_path + '\\' + graph_title + 'binary_waveform' + '.png', dpi=300)
    plt.close(fig)
    time.sleep(1)
    
    return binary_waveform,moving_state_percentage
    
    #if it can compare obujectA and B,do so
    
def normalization(data,shoulder_width):
    return data/shoulder_width
    


read_pickle = pd.read_pickle(file_path +"tracking_results.pkl")
data_0=read_pickle['0'].values.tolist()
data_0=np.array(data_0)

data_0=LinearInterpolationFunction(data_0)

# Compute_PowerSpectrum_Welch(data_0[:,4,0], fs=30, graph_title="data_0[:,4,0]",save_path= file_path)#data, fs=30, graph_title="", save_path=""

# #確信度と座標の配列の位置がずれることに注意
# #Note the shift in the position of the array of beliefs and coordinates.
# data_0[:,4,0]=LowpassFilter(data_0[:,4,0], fs=30, fe=5, n=5)
# data_0[:,4,1]=LowpassFilter(data_0[:,4,1], fs=30, fe=5, n=5)

# after_mov_avg_data_0_x=moving_average(data_0[:,4,0],window_len=15)
# after_mov_avg_data_0_y=moving_average(data_0[:,4,1],window_len=15)

# #肩幅の値で正規化
# x_sholder=list(map(float, data_0[500,5:7,0:1]))
# y_sholder=list(map(float, data_0[500,5:7,1:2]))
# shoulder_point=np.array([abs(x_sholder[1]-x_sholder[0]),abs(y_sholder[1]-y_sholder[0])])
# shoulder_width=np.linalg.norm(shoulder_point, ord=2)
# print(f"shoulder_width:{shoulder_width}")
# # after_mov_avg_data_0_x=normalization(after_mov_avg_data_0_x,shoulder_point[0])
# # after_mov_avg_data_0_y=normalization(after_mov_avg_data_0_y,shoulder_point[1])
# after_mov_avg_data_0_x=normalization(after_mov_avg_data_0_x,1)
# after_mov_avg_data_0_y=normalization(after_mov_avg_data_0_y,1)

#確信度と座標の配列の位置がずれることに注意
#Note the shift in the position of the array of beliefs and coordinates.
# data_0[:,4,0]=LowpassFilter(data_0[:,4,0], fs=30, fe=5, n=5)
# data_0[:,4,1]=LowpassFilter(data_0[:,4,1], fs=30, fe=5, n=5)
data_0[:,4,0]=LowpassFilter(data_0[:,4,0], fs=30, fe=0.1, n=5)
data_0[:,4,1]=LowpassFilter(data_0[:,4,1], fs=30, fe=0.1, n=5)

figure, axis = plt.subplots(2)

axis[0].plot(data_0[:,4,0], label='f(n)', color='red')
axis[0].set_ylabel("X LowpassFilter [pixel]", fontsize=24)
axis[0].tick_params(labelsize=18)
axis[0].grid()

axis[1].plot(data_0[:,4,1], label='f(n)', color='red')
axis[1].set_xlabel("Frame", fontsize=24)
axis[1].set_ylabel("Y LowpassFilter [pixel]", fontsize=24)
axis[1].tick_params(labelsize=18)
axis[1].grid()

plt.show()



#肩幅の値で正規化
x_sholder=list(map(float, data_0[500,5:7,0:1]))
y_sholder=list(map(float, data_0[500,5:7,1:2]))
shoulder_point=np.array([abs(x_sholder[1]-x_sholder[0]),abs(y_sholder[1]-y_sholder[0])])
shoulder_width=np.linalg.norm(shoulder_point, ord=2)
print(f"shoulder_width:{shoulder_width}")
# after_mov_avg_data_0_x=normalization(after_mov_avg_data_0_x,shoulder_point[0])
# after_mov_avg_data_0_y=normalization(after_mov_avg_data_0_y,shoulder_point[1])
# after_mov_avg_data_0_x=normalization(data_0[:,4,0],1)
# after_mov_avg_data_0_y=normalization(data_0[:,4,1],1)
after_mov_avg_data_0_x=normalization(data_0[:,4,0],shoulder_point[0])
after_mov_avg_data_0_y=normalization(data_0[:,4,1],shoulder_point[1])

after_mov_avg_data_0_x=moving_average(after_mov_avg_data_0_x,window_len=150)
after_mov_avg_data_0_y=moving_average(after_mov_avg_data_0_y,window_len=150)

figure, axis = plt.subplots(2)

axis[0].plot(after_mov_avg_data_0_x, label='f(n)', color='red')
axis[0].set_ylabel("X after_mov_avg_data_0_x [pixel]", fontsize=24)
axis[0].tick_params(labelsize=18)
axis[0].grid()

axis[1].plot(after_mov_avg_data_0_y, label='f(n)', color='red')
axis[1].set_xlabel("Frame", fontsize=24)
axis[1].set_ylabel("Y after_mov_avg_data_0_y [pixel]", fontsize=24)
axis[1].tick_params(labelsize=18)
axis[1].grid()

plt.show()



Compute_PowerSpectrum_Welch(after_mov_avg_data_0_x, fs=30, graph_title="after_mov_avg_data_0[0]",save_path= file_path)#data, fs=30, graph_title="", save_path=""
Compute_PowerSpectrum_Welch(after_mov_avg_data_0_y, fs=30, graph_title="after_mov_avg_data_0[1]",save_path= file_path)#data, fs=30, graph_title="", save_path=""


data_0_trajectory=Movement_Trajectory(data=data_0,shoulder_point=shoulder_point,data_x=after_mov_avg_data_0_x,data_y=after_mov_avg_data_0_y,graph_title="data_0_trajectory",save_path= file_path)

data_0_average_speed=average_speed(after_mov_avg_data_0_x,after_mov_avg_data_0_y,fps=30,graph_title="data_0_average_speed",save_path= file_path)





data_0_sample_entropy=sample_entropy(data_0_average_speed)

data_0_confidence_distribution=Confidence_distribution(data_0)

data_0_powerspectrum=Compute_PowerSpectrum_Welch(data_0_average_speed, fs=30, graph_title="data_0_average_speed",save_path= file_path)#data, fs=30, graph_title="", save_path=""

data_0_rms=rms(data_0_average_speed)
print(f"\ndata_0_rms:{data_0_rms}")
    
data_0_binary_waveform,data_0_moving_state_percentage=analysis_still_periods(data_0_average_speed,graph_title="data_0_moving_state_percentage",save_path=file_path)
print(f"data_0_moving_state_percentage:{data_0_moving_state_percentage}")


data_0_sample_entropy=sample_entropy(np.array(data_0_binary_waveform))



#save as dataframe
df = pd.DataFrame(data_0_binary_waveform)
df.to_pickle(file_path + "binary_waveform.pkl")

df = pd.DataFrame(data_0_average_speed)
df.to_pickle(file_path + "Displacement_along_time.pkl")





# data_1=read_pickle['1'].values.tolist()
# data_1=np.array(data_1)
# data_1_trajectory=Movement_Trajectory(data_1)
# data_1_trajectory=Movement_Trajectory(data_1,graph_title="data_1_trajectory",save_path= file_path)


