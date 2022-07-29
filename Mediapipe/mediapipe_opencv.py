import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
from cmath import inf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Path to the videos and name of the video

path = "Videos/Input/videos 17 mayo/"
video_name = "Ruben_12kmh"
extension = "mp4"

cap = cv2.VideoCapture(path+video_name+"."+extension)

# Frame rate of the video

fr = 30

# Number for each joint

Nose = 0
LEyeInner = 1
LEye = 2
LEyeOuter = 3
REyeInner = 4
REye = 5
REyeOuter = 6
LEar = 7
REar = 8
MouthL = 9
MouthR = 10
LShoulder = 11
RShoulder = 12
LElbow = 13
RElbow = 14
LWrist = 15
RWrist = 16
LPinky = 17
RPinky = 18
LIndex = 19
RIndex = 20
LThumb = 21
RThumb = 22
LHip = 23
RHip = 24
LKnee = 25
RKnee = 26
LAnkle = 27
RAnkle = 28
LHeel = 29
RHeel = 30
LBigToe = 31
RBigToe = 32



# equation to calculate angle from 2 vectors

def vectors_to_angle(vector1, vector2) -> float:
    x = np.dot(vector1, -vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    theta = np.degrees(np.arccos(x))
    return theta


# get landmarks and returns a dictionary with the angles of that frame of the video

def landmark_to_angle(lndmks) -> dict:
    
    # coordinates
    Nose_coor = np.array([lndmks[mp_pose.PoseLandmark.NOSE.value].x, lndmks[mp_pose.PoseLandmark.NOSE.value].y])
    LHip_coor = np.array([lndmks[mp_pose.PoseLandmark.LEFT_HIP.value].x, lndmks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    RHip_coor = np.array([lndmks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lndmks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    MidHip_coor = np.array([(LHip_coor[0] + RHip_coor[0])/2, (LHip_coor[1] + RHip_coor[1])/2])
    LKnee_coor = np.array([lndmks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lndmks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
    RKnee_coor = np.array([lndmks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lndmks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
    LAnkle_coor = np.array([lndmks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lndmks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
    RAnkle_coor = np.array([lndmks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lndmks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
    LBigToe_coor = np.array([lndmks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, lndmks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y])
    RBigToe_coor = np.array([lndmks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, lndmks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])
    
    # vectors
    Torso_vector = MidHip_coor - Nose_coor
    Hip_vector = LHip_coor - RHip_coor
    LFemur_vector = LKnee_coor - LHip_coor
    RFemur_vector = RKnee_coor - RHip_coor
    LTibia_vector = LAnkle_coor - LKnee_coor
    RTibia_vector = RAnkle_coor - RKnee_coor
    LFoot_vector = LBigToe_coor - LAnkle_coor
    RFoot_vector = RBigToe_coor - RAnkle_coor
    
    # angles
    TorsoLHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
    TorsoRHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
    LHip_angle = vectors_to_angle(LFemur_vector, Hip_vector)
    RHip_angle = vectors_to_angle(RFemur_vector, -Hip_vector)
    LKnee_angle = vectors_to_angle(LTibia_vector, LFemur_vector)
    RKnee_angle = vectors_to_angle(RTibia_vector, RFemur_vector)
    LAnkle_angle = vectors_to_angle(LFoot_vector, LTibia_vector)
    RAnkle_angle = vectors_to_angle(RFoot_vector, RTibia_vector)
    
    
    dict_angles = {"TorsoLHip_angle": TorsoLHip_angle, "TorsoRHip_angle": TorsoRHip_angle, "LHip_angle": LHip_angle,
                   "RHip_angle": RHip_angle, "LKnee_angle": LKnee_angle, "RKnee_angle": RKnee_angle, "LAnkle_angle": LAnkle_angle, "RAnkle_angle": RAnkle_angle}
    return dict_angles



# Plot the angles

def plot_angles(df) -> None:
    
    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(3, 3, figsize=(24, 14))
        
        fig.suptitle("Evolución en el tiempo de los ángulos de las articulaciones")
        
        sns.lineplot(ax = axes[0, 0], data = df, x = "Time_in_sec", y = "TorsoLHip_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Torso con parte izquierda de la cadera (º)")
        sns.lineplot(ax = axes[0, 1], data = df, x = "Time_in_sec", y = "TorsoRHip_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Torso con parte derecha de la cadera (º)")
        sns.lineplot(ax = axes[1, 0], data = df, x = "Time_in_sec", y = "LHip_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Cadera con la pierna izquierda (º)")
        sns.lineplot(ax = axes[1, 1], data = df, x = "Time_in_sec", y = "RHip_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Cadera con la pierna derecha (º)")
        sns.lineplot(ax = axes[1, 2], data = df, x = "Time_in_sec", y = "LKnee_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Rodilla izquierda (º)")
        sns.lineplot(ax = axes[2, 0], data = df, x = "Time_in_sec", y = "RKnee_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Rodilla derecha (º)")
        sns.lineplot(ax = axes[2, 1], data = df, x = "Time_in_sec", y = "LAnkle_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Tobillo izquierdo (º)")
        sns.lineplot(ax = axes[2, 2], data = df, x = "Time_in_sec", y = "RAnkle_angle").set(xlabel = "Tiempo (segundos)", ylabel = "Tobillo derecho (º)")
        
        plt.show()


# Fourier transform

def fourier_period(y) -> float:
    # Number of sample points
    N = len(y)
    # Sample spacing
    T = 1.0 / fr
    
    yf = np.fft.fft(y)
    yf_abs = 2.0/N * np.abs(yf[0:N//2])
    xf = np.fft.fftfreq(N, T)[:N//2]
    yf_max = np.copy(yf_abs)
    yf_max[0] = 0.0
    period = xf[np.argmax(yf_max)]
    
    # plt.plot(xf, yf_abs)
    # plt.xlim(right = 4 * period)
    # plt.axvline(period, color='red')
    # plt.xlabel('Tiempo (segundos)')
    # plt.title("Transformada rápida de Fourier")
    # plt.show()
    
    return period


# Autocorrelation

def autocorrelation(y, t) -> float:
    s = pd.Series(y) 
    
    x = sm.tsa.acf(s)
    
    plt.plot(t[:len(x)], x)
    plt.grid()
    plt.show()
    
    xf = np.copy(x)
    xf[0] = 0.0
    period = t[np.argmax(xf)]
    
    return period
    


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
                if (y.iloc[i_max[j]] - y.iloc[i_min[i-1]]) > 60:
                    i_s.append(i_min[i-1])
    
    splits = y.iloc[i_s]
    
    # make sure we do not have two local minimums in less than 0.3 seconds.
    drop_index = []
    
    for loc, _ in enumerate(splits):
        if loc == 0:
            continue
        
        if (loc > 1):
            if (splits.index[loc-1] in drop_index):
                if (splits.index[loc-1] in drop_index) and (((splits.index[loc] - splits.index[loc-2]) >= min_dis) and (splits.index[loc] - splits.index[loc-2]) <= max_dis):
                    continue
        
        if ((splits.index[loc] - splits.index[loc-1]) < min_dis) or (splits.index[loc] - splits.index[loc-1]) > max_dis:
            drop_index.append(splits.index[loc])
        
        
    
    # split the data in each cycle
    
    # print(splits)
    # print(drop_index)
    
    splits = splits.drop(drop_index).index
    
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
        y_s_maxs = y_s_maxs[y_s_maxs.values>135]
        #print(y2_s_maxs)
        #if y2_s_maxs.index[0] < 150:
        #    print(y2_s)
        #    print(y2_s_maxs)
        #    print(y2_s_maxs.index)
        
        first_contact = min(y_s_maxs.index)
        
        launching = max(y_s_maxs.index) + 1
        
        # print("First contact: ", first_contact)
        # print("Launching: ", launching)
        
        # get the moment when the knee angle is in its minimum
        #launching  = y2_s.index[minimum(y2_s)[1]]
        
        t = (1.0/fr) * (launching - first_contact)
        
        if (t > 0.1) & (t < 2*period/3):
            t_diffs.append(t)
    
    t_diff = np.mean(t_diffs)
    
    return t_diff





# Main function

def main() -> None:
    
    list_of_dicts = []
    
    with mp_pose.Pose(static_image_mode=False) as pose:
        
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            
            try:
                landmarks = results.pose_landmarks.landmark
                list_of_dicts.append(landmark_to_angle(landmarks))
            except:
                pass
            
            scale_percent = 50 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # cv2.imshow("Mediapipe", cv2.resize(frame, dim, interpolation = cv2.INTER_AREA))
            if (cv2.waitKey(1) & 0xFF == 27) | (cv2.waitKey(1) & 0xFF == ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    
    df_angles = pd.DataFrame(list_of_dicts)
    df_angles["Time_in_sec"] = [n/fr for n in range(len(df_angles))]
    
    
    df_to_plot = df_angles.loc[(df_angles["Time_in_sec"] >= 0) & (df_angles["Time_in_sec"] <= 5)]
    
    plot_angles(df_to_plot)
    
    period = autocorrelation(df_angles["RHip_angle"], df_angles["Time_in_sec"])
    print("Periodo entre cada paso:", period/2, "segundos")
    print("Ratio de pasos:", (2*60.0)/period, "pasos/minuto")
    
    contact_time_r = contact(df_angles["RKnee_angle"], fr, period, min_dis=int(period*fr)-5, max_dis=int(period*fr)+5)
    print("Tiempo de contacto en el suelo del pie derecho:", contact_time_r, "s")
    contact_time_l = contact(df_angles["LKnee_angle"], fr, period, min_dis=int(period*fr)-5, max_dis=int(period*fr)+5)
    print("Tiempo de contacto en el suelo del pie izquierdo:", contact_time_l, "s")
    
    flight_ratio_r = 100 * (1 - (2 * contact_time_r / period))
    print("Ratio de tiempo en el aire del pie derecho:", flight_ratio_r, "%")
    flight_ratio_l = 100 * (1 - (2 * contact_time_l / period))
    print("Ratio de tiempo en el aire del pie izquierdo:", flight_ratio_l, "%")
    
    


if __name__ == '__main__':
    main()