import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
import datetime
import re
# import TimeLib

import pandas as pd
# file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/ri_shouki/0947-0957/cleaned/"
file_path = "/home/user/Users/kai/data/pose_results/minamiyoshida_1012/takano_nanami/1000-1010/cleaned/"
# file_path = "/home/user/Users/kai/data/pose_results/20220627_5th_period/first_v2/cleaned/"


def LowpassFilter(wave, fs, fe, n):
    wave=list(map(float, wave))
    
    nyq = fs / 2.0
    b, a = signal.butter(1, fe / nyq, btype='low')
    for i in range(0, n):
        wave = signal.filtfilt(b, a, wave)
    return np.array(wave)

'''
関数：パワースペクトルを求める
引数：data（信号データ）, fs（サンプリング周波数）, graph_data（グラフタイトル）, save_path（グラフの保存パス）
戻値：np.mean(np.sqrt(Pxx_den_med))（パワースペクトルの平均値）
Function: Find the power spectrum
Arguments: data (signal data), fs (sampling frequency), graph_data (graph title), save_path (graph save path)
Return: np.mean(np.sqrt(Pxx_den_med)) (mean value of power spectrum)
'''

def Compute_PowerSpectrum_Welch(data, fs=30, graph_title="", save_path=""):
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
    plt.ylabel("Y coordinate [pixel]", fontsize=24)
    plt.xticks(fontsize=18)
    plt.xticks(fontsize=18)
    #plt.ylim(0, 15)
    plt.tick_params(labelsize=18)
    plt.grid()
    plt.show()

    # 周波数信号(元)
    #plt.subplot(212)
    #plt.plot(freq, np.abs(F), label='|F(k)|')
    #plt.xlabel('Frequency', fontsize=12)
    #plt.ylabel('Amplitude', fontsize=12)
    #plt.grid()
    #leg = plt.legend(loc='lower center', bbox_to_anchor=(1, 1), fontsize=15)
    #leg.get_frame().set_alpha(1)

    # パワースペクトル
    plt.subplot(212)
    plt.semilogy(f_med, np.sqrt(Pxx_den_med), color='blue')
    plt.xlabel('Frequency [Hz]', fontsize=24)
    plt.ylabel('Linear spectrum [V RMS]', fontsize=24)
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

read_pickle = pd.read_pickle(file_path +"tracking_results.pkl")
data_0=read_pickle['0'].values.tolist()
data_0=np.array(data_0)
data_0[:,4,0]=LowpassFilter(data_0[:,4,0], fs=30, fe=1, n=5)
data_0[:,4,1]=LowpassFilter(data_0[:,4,1], fs=30, fe=1, n=5)
id_0_centroid_y_keypoints=data_0[:,-1,1:2]
id_0_centroid_y_keypoints=map(float, id_0_centroid_y_keypoints)
id_0_centroid_y_keypoints=list(id_0_centroid_y_keypoints)
Compute_PowerSpectrum_Welch(id_0_centroid_y_keypoints, fs=30, graph_title="id_0_centroid_y_keypoints",save_path= file_path)#data, fs=30, graph_title="", save_path=""

id_0_centroid_x_keypoints=list(map(float, data_0[:,-1,0:1]))
Compute_PowerSpectrum_Welch(id_0_centroid_x_keypoints, fs=30, graph_title="id_0_centroid_x_keypoints",save_path= file_path)#data, fs=30, graph_title="", save_path=""

id_0_left_ear_x_keypoints=list(map(float, data_0[:,3,0:1]))
Compute_PowerSpectrum_Welch(id_0_left_ear_x_keypoints, fs=30, graph_title="id_0_left_ear_x_keypoints",save_path= file_path)#data, fs=30, graph_title="", save_path=""
id_0_left_ear_y_keypoints=list(map(float, data_0[:,3,1:2]))
Compute_PowerSpectrum_Welch(id_0_left_ear_y_keypoints, fs=30, graph_title="id_0_left_ear_y_keypoints",save_path= file_path)#data, fs=30, graph_title="", save_path=""

data_1=np.array(read_pickle['1'].values.tolist())
id_1_centroid_x_keypoints=list(map(float, data_1[:,-1,0:1]))
id_1_centroid_y_keypoints=list(map(float, data_1[:,-1,1:2]))
Compute_PowerSpectrum_Welch(id_1_centroid_x_keypoints, fs=30, graph_title="id_1_centroid_x_keypoints",save_path= file_path)#data, fs=30, graph_title="", save_path=""
Compute_PowerSpectrum_Welch(id_1_centroid_y_keypoints, fs=30, graph_title="id_1_centroid_y_keypoints",save_path= file_path)#data, fs=30, graph_title="", save_path=""

# data_2=np.array(read_pickle['2'].values.tolist())
# id_2_centroid_x_keypoints=list(map(float, data_1[:,-1,0:1]))
# id_2_centroid_y_keypoints=list(map(float, data_1[:,-1,1:2]))
# Compute_PowerSpectrum_Welch(id_2_centroid_x_keypoints, fs=30, graph_title="id_2_centroid_x_keypoints",save_path= file_path)#data, fs=30, graph_title="", save_path=""
# Compute_PowerSpectrum_Welch(id_2_centroid_y_keypoints, fs=30, graph_title="id_2_centroid_y_keypoints",save_path= file_path)#data, fs=30, graph_title="", save_path=""