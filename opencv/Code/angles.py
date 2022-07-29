from cmath import inf
import os, json
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm


# Path to the videos and name of the video

path = "Videos/Output_Json/"
video_name = "Sujeto1_R1_10kmh_20210715_175707"

print("Video name: ", video_name, "\n")


# Frame rate of the video

fr = 30


# Number for each joint

Nose = 0
Neck = 1
RShoulder = 2
RElbow = 3
RWrist = 4
LShoulder = 5
LElbow = 6
LWrist = 7
MidHip = 8
RHip = 9
RKnee = 10
RAnkle = 11
LHip = 12
LKnee = 13
LAnkle = 14
REye = 15
LEye = 16
REar = 17
LEar = 18
LBigToe = 19
LSmallToe = 20
LHeel = 21
RBigToe = 22
RSmallToe = 23
RHeel = 24
Background = 25


# equation to calculate angle from 2 vectors

def vectors_to_angle(vector1, vector2) -> float:
    x = np.dot(vector1, -vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    theta = np.degrees(np.arccos(x))
    return theta


# get a json file and returns a dictionary with the angles of that frame of the video
def json_to_angle(json_name) -> dict:
    with open(path+video_name+"/"+json_name) as json_file:
        
        data = json.load(json_file)['people'][0]['pose_keypoints_2d']
        
        
        # coordinates
        Neck_coor = np.array([data[Neck*3], data[Neck*3+1]])
        RShoulder_coor = np.array([data[RShoulder*3], data[RShoulder*3+1]])
        LShoulder_coor = np.array([data[LShoulder*3], data[LShoulder*3+1]])
        MidHip_coor = np.array([data[MidHip*3], data[MidHip*3+1]])
        LHip_coor = np.array([data[LHip*3], data[LHip*3+1]])
        RHip_coor = np.array([data[RHip*3], data[RHip*3+1]])
        LKnee_coor = np.array([data[LKnee*3], data[LKnee*3+1]])
        RKnee_coor = np.array([data[RKnee*3], data[RKnee*3+1]])
        LAnkle_coor = np.array([data[LAnkle*3], data[LAnkle*3+1]])
        RAnkle_coor = np.array([data[RAnkle*3], data[RAnkle*3+1]])
        LBigToe_coor = np.array([data[LBigToe*3], data[LBigToe*3+1]])
        RBigToe_coor = np.array([data[RBigToe*3], data[RBigToe*3+1]])
        
        
        # vectors
        Torso_vector = MidHip_coor - Neck_coor
        RTorso_vector = RHip_coor - RShoulder_coor
        LTorso_vector = LHip_coor - LShoulder_coor
        Hip_vector = LHip_coor - RHip_coor
        LFemur_vector = LKnee_coor - LHip_coor
        RFemur_vector = RKnee_coor - RHip_coor
        LTibia_vector = LAnkle_coor - LKnee_coor
        RTibia_vector = RAnkle_coor - RKnee_coor
        LFoot_vector = LBigToe_coor - LAnkle_coor
        RFoot_vector = RBigToe_coor - RAnkle_coor
        
        LFemur_vector_perp = LFemur_vector.copy()
        LFemur_vector_perp[1] = LFemur_vector[0]
        LFemur_vector_perp[0] = -LFemur_vector[1]
        
        RFemur_vector_perp = RFemur_vector.copy()
        RFemur_vector_perp[1] = RFemur_vector[0]
        RFemur_vector_perp[0] = -RFemur_vector[1]
        
        # angles
        TorsoHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
        LShoulderFemur_angle = vectors_to_angle(LFemur_vector_perp, LTorso_vector) + 90
        RShoulderFemur_angle = vectors_to_angle(RFemur_vector_perp, RTorso_vector) + 90
        LHip_angle = vectors_to_angle(LFemur_vector, Hip_vector)
        RHip_angle = vectors_to_angle(RFemur_vector, Hip_vector)
        LKnee_angle = vectors_to_angle(LTibia_vector, LFemur_vector_perp) + 90
        RKnee_angle = vectors_to_angle(RTibia_vector, RFemur_vector_perp) + 90
        LAnkle_angle = vectors_to_angle(LFoot_vector, LTibia_vector)
        RAnkle_angle = vectors_to_angle(RFoot_vector, RTibia_vector)
        
        
        dict_angles = {"TorsoHip_angle": TorsoHip_angle, "LShoulderFemur_angle": LShoulderFemur_angle, "RShoulderFemur_angle": RShoulderFemur_angle, 
                       "LHip_angle": LHip_angle, "RHip_angle": RHip_angle, "LKnee_angle": LKnee_angle, "RKnee_angle": RKnee_angle, 
                       "LAnkle_angle": LAnkle_angle, "RAnkle_angle": RAnkle_angle}
        return dict_angles


# Plot the angles

def plot_angles(df) -> None:
    
    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(5, 2, figsize=(9, 30))
        
        fig.suptitle("Evolución en el tiempo de los ángulos de las articulaciones")
        
        sns.lineplot(ax = axes[0, 0], data = df, x = "Time_in_sec", y = "TorsoHip_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Torso/cadera (º)")
        sns.lineplot(ax = axes[1, 0], data = df, x = "Time_in_sec", y = "LShoulderFemur_angle").set(xlabel = "Tiempo (segundos)", 
                                                                                                    ylabel = "Hombro/rodilla izq. (º)")
        sns.lineplot(ax = axes[1, 1], data = df, x = "Time_in_sec", y = "RShoulderFemur_angle").set(xlabel = "Tiempo (segundos)", 
                                                                                                    ylabel = "Hombro/rodilla der. (º)")
        sns.lineplot(ax = axes[2, 0], data = df, x = "Time_in_sec", y = "LHip_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Cadera/pierna izq. (º)")
        sns.lineplot(ax = axes[2, 1], data = df, x = "Time_in_sec", y = "RHip_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Cadera/pierna der. (º)")
        sns.lineplot(ax = axes[3, 0], data = df, x = "Time_in_sec", y = "LKnee_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Rodilla izq. (º)")
        sns.lineplot(ax = axes[3, 1], data = df, x = "Time_in_sec", y = "RKnee_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Rodilla der. (º)")
        sns.lineplot(ax = axes[4, 0], data = df, x = "Time_in_sec", y = "LAnkle_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Tobillo izq. (º)")
        sns.lineplot(ax = axes[4, 1], data = df, x = "Time_in_sec", y = "RAnkle_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Tobillo der. (º)")
        
        plt.show()
        
        sns.set(font_scale = 1.5)
        sns.lineplot(data = df, x = "Time_in_sec", y = "LKnee_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Rodilla izq. (º)")
        
        plt.show()


# # Fourier transform

# def fourier_period(y) -> float:
#     # Number of sample points
#     N = len(y)
#     # Sample spacing
#     T = 1.0 / fr
    
#     yf = np.fft.fft(y)
#     yf_abs = 2.0/N * np.abs(yf[0:N//2])
#     xf = np.fft.fftfreq(N, T)[:N//2]
#     yf_max = np.copy(yf_abs)
#     yf_max[0] = 0.0
#     period = xf[np.argmax(yf_max)]
    
#     plt.plot(xf, yf_abs)
#     plt.xlim(right = 4 * period)
#     plt.axvline(period, color='red')
#     plt.xlabel('Tiempo (segundos)')
#     plt.title("Transformada rápida de Fourier")
#     plt.show()
    
#     return period


# Autocorrelation

def autocorrelation(y, t) -> float:
    s = pd.Series(y) 
    
    x = sm.tsa.acf(s, missing = "drop")
    
    plt.plot(t[:len(x)], x)
    plt.grid()
    plt.show()
    
    xf = np.copy(x)
    xf[0] = 0.0
    xf[1] = 0.0
    xf[2] = 0.0
    period = t[np.argmax(xf)]
    
    return period
    


# The contact time starts with the leg have an local maximum in the angle due to the impact, at the same time the knee has his first angle maximum.
# The contact time finishes with the maximum angle of the leg and the second maximum of the knee.

# Get two largest values in a list

# def twoLargest(numbers) -> list:
#     x=0
#     y=0
#     x_loc=0
#     y_loc=0
#     i = 0
#     for loc, n in enumerate(numbers):
#         i+=1
#         if n > x:
#             y = x
#             y_loc = x_loc
#             x = n
#             x_loc = loc
#             continue
#         if n > y:
#             y = n
#             y_loc = loc

#     return [x, y], [x_loc, y_loc]


def maximum(numbers) -> list:
    z = 0
    z_loc = 0
    for loc, n in enumerate(numbers):
        if n > z:
            z = n
            z_loc = loc
    
    return z, z_loc

def minimum(numbers) -> list:
    z = inf
    z_loc = 0
    for loc, n in enumerate(numbers):
        if n < z:
            z = n
            z_loc = loc
    
    return z, z_loc
    

#old version

# def contact(y, fr) -> float:
#     # a cycle can be determined by a local minimum lower than 120º
    
#     splits = argrelextrema(y.values, np.less)[0][y[argrelextrema(y.values, np.less)[0]]<120]
    
#     t_diffs = []
    
#     for i, s in enumerate(splits):
#         if i>0:
#             if s-splits[i-1]<10:
#                 continue
#             y_s = y[splits[i-1]:s]
            
#             _ , max_locs = twoLargest(y_s.values[argrelextrema(y_s.values, np.greater)[0]])
#             max_ilocs = argrelextrema(y_s.values, np.greater)[0][max_locs]
            
#             t_diffs.append((1.0/fr) * abs(max_ilocs[0] - max_ilocs[1]))
    
#     t_diff = np.median(t_diffs)
    
#     return t_diff

#new version

# def contact(y, y2, fr) -> float:
#     # a cycle can be determined by a local minimum lower than 120º
    
#     splits = argrelextrema(y.values, np.less)[0][y2[argrelextrema(y.values, np.less)[0]]<120]
    
#     t_diffs = []
    
#     for i, s in enumerate(splits):
#         if i>0:
#             if s-splits[i-1]<10:
#                 continue
#             y_s = y[splits[i-1]:s]
            
#             _ , max_locs = twoLargest(y_s.values[argrelextrema(y_s.values, np.greater)[0]])
#             max_ilocs = argrelextrema(y_s.values, np.greater)[0][max_locs]
#             #max_iloc = min(max_ilocs)
            
#             y2_s = y2[splits[i-1]:s]
            
#             _ , max_iloc = maximum(y2_s)
#             _ , min_iloc = minimum(y2_s)
            
            
#             t = (1.0/fr) * abs(max_iloc - min_iloc)
            
#             t_diffs.append(t)
    
#     t_diff = np.mean(t_diffs)
    
    
#     return t_diff


def contact(y, fr, period, min_dis=9, max_dis=30) -> float:
    # get the local minimums that are less than 150
    i_min = argrelextrema(y.values, np.less)[0]
    i_max = argrelextrema(y.values, np.greater)[0]
    
    i_s = []
    
    for i, _ in enumerate(i_min):
        for j, _ in enumerate(i_max):
            if i == 0:
                continue
            if (i_min[i] - i_min[i-1]) > (i_max[j] - i_min[i-1]) and (i_max[j] - i_min[i-1]) > 0:
                if (y.iloc[i_max[j]] - y.iloc[i_min[i-1]]) > 40:
                    i_s.append(i_min[i-1])
    
    splits = y.iloc[i_s]
    
    # make sure we do not have two local minimums in less than 0.3 seconds.
    drop_index = []
    
    for loc, _ in enumerate(splits):
        if loc == 0:
            continue
        
        if ((splits.index[loc] - splits.index[loc-1]) < min_dis) or (splits.index[loc] - splits.index[loc-1]) > max_dis:
            drop_index.append(splits.index[loc])
    
    # split the data in each cycle
    
    splits = splits.drop(drop_index).index
    
    # print(splits)
    
    t_diffs = []
    
    for i, s in enumerate(splits):
        
        if s == splits[0]:
            continue
        
        #y1_s = y1[splits[i-1]:s]
        
        # get the moment when the angle of the knee with the torso goes to greater than 180º
        #try:
        #    y1_s_180 = min(y1_s.index[y1_s>170].to_list())
        #except:
        #    continue
        
        # this is the moment where the foot do the first contact with a 0 angle between the knee and the torso
        
        #first_contact = y1_s_180 - 1
        
        #launching = y1_s.index[maximum(y1_s)[1]]
        
        y_s = y[splits[i-1]:s]
        
        j_max = argrelextrema(y_s.values, np.greater)
        y_s_maxs = y_s.iloc[j_max]
        y_s_maxs = y_s_maxs[y_s_maxs.values>150]
        #print(y2_s_maxs)
        #if y2_s_maxs.index[0] < 150:
        #    print(y2_s)
        #    print(y2_s_maxs)
        #    print(y2_s_maxs.index)
        
        first_contact = min(y_s_maxs.index)
        
        launching = max(y_s_maxs.index)
        
        # print("Max dis: ", max_dis)
        # print("First contact: ", first_contact)
        # print("Launching: ", launching)
        
        # get the moment when the knee angle is in its minimum
        #launching  = y2_s.index[minimum(y2_s)[1]]
        
        t = (1.0/fr) * (launching - first_contact)
        
        if (t > 0.1) & (t < 2*period/3):
            t_diffs.append(t)
    
    # print(t_diffs)
    
    t_diff = np.mean(t_diffs)
    
    return t_diff


# Main function

def main() -> None:
    
    json_files = [pos_json for pos_json in os.listdir(path+video_name) if pos_json.endswith('.json')]
    
    list_of_dicts = []
    for json_name in json_files[:len(json_files)-1]:
        try:
            list_of_dicts.append(json_to_angle(json_name))
        except:
            pass
    
    df_angles = pd.DataFrame(list_of_dicts)
    df_angles["Time_in_sec"] = [n/fr for n in range(len(df_angles))]
    
    # print(df_angles)
    
    df_to_plot = df_angles.loc[(df_angles["Time_in_sec"] >= 0) & (df_angles["Time_in_sec"] <= 5)]
    
    plot_angles(df_to_plot)
    
    period = autocorrelation(df_angles["RHip_angle"], df_angles["Time_in_sec"])
    print("Periodo entre cada paso:", period/2, "segundos")
    print("Ratio de pasos:", (2*60.0)/period, "pasos/minuto")
                                                                                                                                                                                                                                                      
    contact_time_r = contact(df_angles["RKnee_angle"], fr, period, min_dis=int(period*fr)-5, max_dis=int(period*fr)+5)
    contact_time_l = contact(df_angles["LKnee_angle"], fr, period, min_dis=int(period*fr)-5, max_dis=int(period*fr)+5)
    print("Tiempo de contacto en el suelo del pie izquierdo:", contact_time_l, "s")
    print("Tiempo de contacto en el suelo del pie derecho:", contact_time_r, "s")
    
    flight_ratio_r = 100 * (1 - (2 * contact_time_r / period))
    flight_ratio_l = 100 * (1 - (2 * contact_time_l / period))
    print("Ratio de tiempo en el aire del pie izquierdo:", flight_ratio_l, "%")
    print("Ratio de tiempo en el aire del pie derecho:", flight_ratio_r, "%")
    

if __name__ == '__main__':
    main()