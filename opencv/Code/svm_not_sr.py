from cmath import inf
import os, json
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Path to the videos and name of the video

path = "Videos/Output_Json/"

train_video_names = ["Sujeto1_R1_10kmh_20210715_175707", "Sujeto1_R2_12kmh_20210715_175812", "Sujeto2_R1_10kmh_20210715_180231", "Sujeto2_R2_12kmh_20210715_180326", 
               "Sujeto4_R1_10kmh_20210715_181148", "Sujeto5_R1_10kmh_20210715_181749", "Sujeto5_R2_12kmh_20210715_181839", "Sujeto6_R1_10kmh_20210715_182209", 
               "Sujeto6_R2_12kmh_20210715_182300", "Sujeto7_R2_12kmh_20210715_182813", "Sujeto8_R1_10kmh_20210715_183143",
               "Sujeto9_R1_10kmh_20210715_175119", "Sujeto9_R2_12kmh_20210715_175353"]
test_video_names = ["Sujeto3_R1_10kmh_20210715_180802", "Sujeto4_R2_12kmh_20210715_181312", "Sujeto7_R1_10kmh_20210715_182723", "Sujeto8_R2_12kmh_20210715_183236"]

metrics_path = "Videos/Metrics.csv"

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
def json_to_angle(json_name, video_name) -> dict:
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
        
        # angles
        TorsoHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
        LShoulderFemur_angle = vectors_to_angle(LFemur_vector, LTorso_vector)
        RShoulderFemur_angle = vectors_to_angle(RFemur_vector, RTorso_vector)
        LHip_angle = vectors_to_angle(LFemur_vector, Hip_vector)
        RHip_angle = vectors_to_angle(RFemur_vector, Hip_vector)
        LKnee_angle = vectors_to_angle(LTibia_vector, LFemur_vector)
        RKnee_angle = vectors_to_angle(RTibia_vector, RFemur_vector)
        LAnkle_angle = vectors_to_angle(LFoot_vector, LTibia_vector)
        RAnkle_angle = vectors_to_angle(RFoot_vector, RTibia_vector)
        
        
        dict_angles = {"TorsoHip_angle": TorsoHip_angle, "LShoulderFemur_angle": LShoulderFemur_angle, "RShoulderFemur_angle": RShoulderFemur_angle, 
                       "LHip_angle": LHip_angle, "RHip_angle": RHip_angle, "LKnee_angle": LKnee_angle, "RKnee_angle": RKnee_angle, 
                       "LAnkle_angle": LAnkle_angle, "RAnkle_angle": RAnkle_angle}
        return dict_angles


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

def autocorrelation(y, t) -> float:
    s = pd.Series(y) 
    
    x = sm.tsa.acf(s)
    xf = np.copy(x)
    xf[0] = 0.0
    period = t[np.argmax(xf)]
    
    return period

def df_slice(df, df_m, v_name, min_dis=9, max_dis=30):
    y1 = df["LShoulderFemur_angle"]
    i_min = argrelextrema(y1.values, np.less)
    splits = y1.iloc[i_min]
    splits = splits[splits.values<160]
    
    # make sure we do not have two local minimums in less than 0.3 seconds.
    drop_index = []
    
    for loc, _ in enumerate(splits):
        
        if loc == 0:
            continue
        
        if (splits.index[loc] - splits.index[loc-1]) < min_dis:
            drop_index.append(splits.index[loc])
    
    # split the data in each cycle
    
    splits = splits.drop(drop_index).index
    
    for i, s in enumerate(splits):
        
        if s == splits[0]:
            
            column_names = []
            for col in df.columns:
                for j in range(max_dis):
                    column_names.append(col+"_"+str(j))
            
            wider_df = pd.DataFrame(columns=column_names)
            
            continue
        
        if (s - splits[i-1]) < max_dis:
            row_i = []
            for c in df.columns:
                for k in range(max_dis):
                    if k < (s - splits[i-1]):
                        row_i.append(df.loc[splits[i-1]+k, c])
                    else:
                        row_i.append(np.mean(df.loc[splits[i-1]:s, c]))

            wider_df.loc[len(wider_df)] = row_i
    row_j = df_m[df_m["Video"] == v_name].values.flatten().tolist()
    
    for l, col_i in enumerate(df_m.columns):
        if col_i == "Video":
            continue
        wider_df[col_i] = row_j[l]
    
    return wider_df


def train_test(df, output_cols, test_size=0.3) :
    
    Y = df[output_cols]
    X = df.drop(columns=output_cols)
    X = X.drop(columns=["Video", "Step rate", "Step rate pred"])
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    
    return X_train, X_test, Y_train, Y_test


def train_svm(X, Y, output):
    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr.fit(X, Y[[output]].values.ravel())
    return regr


def print_results(X_train, X_test, Y_train, Y_test, ctl_model, ctr_model, frl_model, frr_model) -> None:
    
    print("\n\nContact time L")
    print("\nTrain")
    print("R^2 =", ctl_model.score(X_train, Y_train[["Contact time L"]].values.ravel()))
    print("RMSE =", np.sqrt(mean_squared_error(Y_train[["Contact time L"]].values.ravel(), ctl_model.predict(X_train))))
    print("\nTest")
    print("R^2 =", ctl_model.score(X_test, Y_test[["Contact time L"]].values.ravel()))
    print("RMSE =", np.sqrt(mean_squared_error(Y_test[["Contact time L"]].values.ravel(), ctl_model.predict(X_test))))
    
    print("\n\nContact time R")
    print("\nTrain")
    print("R^2 =", ctr_model.score(X_train, Y_train[["Contact time R"]].values.ravel()))
    print("RMSE =", np.sqrt(mean_squared_error(Y_train[["Contact time R"]].values.ravel(), ctr_model.predict(X_train))))
    print("\nTest")
    print("R^2 =", ctr_model.score(X_test, Y_test[["Contact time R"]].values.ravel()))
    print("RMSE =", np.sqrt(mean_squared_error(Y_test[["Contact time R"]].values.ravel(), ctr_model.predict(X_test))))
    
    print("\n\nFlight Ratio L")
    print("\nTrain")
    print("R^2 =", frl_model.score(X_train, Y_train[["Flight Ratio L"]].values.ravel()))
    print("RMSE =", np.sqrt(mean_squared_error(Y_train[["Flight Ratio L"]].values.ravel(), frl_model.predict(X_train))))
    print("\nTest")
    print("R^2 =", frl_model.score(X_test, Y_test[["Flight Ratio L"]].values.ravel()))
    print("RMSE =", np.sqrt(mean_squared_error(Y_test[["Flight Ratio L"]].values.ravel(), frl_model.predict(X_test))))
    
    print("\n\nFlight Ratio R")
    print("\nTrain")
    print("R^2 =", frr_model.score(X_train, Y_train[["Flight Ratio R"]].values.ravel()))
    print("RMSE =", np.sqrt(mean_squared_error(Y_train[["Flight Ratio R"]].values.ravel(), frr_model.predict(X_train))))
    print("\nTest")
    print("R^2 =", frr_model.score(X_test, Y_test[["Flight Ratio R"]].values.ravel()))
    print("RMSE =", np.sqrt(mean_squared_error(Y_test[["Flight Ratio R"]].values.ravel(), frr_model.predict(X_test))))
    

def predict_videos(df, df_output, output_col, model):
    
    dropped_cols = ["Step rate", "Contact time L", "Contact time R", "Flight Ratio L", "Flight Ratio R", "Step rate pred", "Video"]
    X = df.drop(columns=dropped_cols)
    
    df_output[output_col+" pred"] = model.predict(X)
    
    return df_output





# Main function

def main() -> None:
    
    print("\n\nLoading Training Videos:")
    
    video_i = 1
    
    for video_name in train_video_names:
        
        print(video_i, " // ", len(train_video_names))
        
        video_i += 1
        
        json_files = [pos_json for pos_json in os.listdir(path+video_name) if pos_json.endswith('.json')]
    
        list_of_dicts = []
        for json_name in json_files[:len(json_files)-1]:
            try:
                list_of_dicts.append(json_to_angle(json_name, video_name))
            except:
                pass
        
        df_angles = pd.DataFrame(list_of_dicts)
        
        df_angles["Time_in_sec"] = [n/fr for n in range(len(df_angles))]
        period = autocorrelation(df_angles["RHip_angle"], df_angles["Time_in_sec"])
        df_angles.drop("Time_in_sec", axis=1, inplace=True)
        
        # Normalize columns
        #df_angles = (df_angles-df_angles.mean())/df_angles.std()
        
        df_metrics = pd.read_csv(metrics_path, sep=";")
        sliced_df = df_slice(df_angles, df_metrics, video_name)
        sliced_df["Video"] = video_name
        sliced_df["Step rate pred"] = (2*60.0)/period
        
        if video_name == train_video_names[0]:
            df_complete = sliced_df.copy()
        else:
            df_complete = pd.concat([df_complete, sliced_df])
    
    output_cols = ["Contact time L", "Contact time R", "Flight Ratio L", "Flight Ratio R"]
    
    X_train, X_test, Y_train, Y_test = train_test(df_complete, output_cols)
    
    ctl_model = train_svm(X_train, Y_train, "Contact time L")
    ctr_model = train_svm(X_train, Y_train, "Contact time R")
    frl_model = train_svm(X_train, Y_train, "Flight Ratio L")
    frr_model = train_svm(X_train, Y_train, "Flight Ratio R")
    
    print("\n\nTraining results:")
    print_results(X_train, X_test, Y_train, Y_test, ctl_model, ctr_model, frl_model, frr_model)

    
    print("\n\nLoading Test Videos:")
    
    video_j = 1

    for video_name in test_video_names:
        
        print(video_j, " // ", len(test_video_names))
        
        video_j += 1
        
        json_files = [pos_json for pos_json in os.listdir(path+video_name) if pos_json.endswith('.json')]
    
        list_of_dicts = []
        for json_name in json_files[:len(json_files)-1]:
            try:
                list_of_dicts.append(json_to_angle(json_name, video_name))
            except:
                pass
        
        df_angles = pd.DataFrame(list_of_dicts)
        
        df_angles["Time_in_sec"] = [n/fr for n in range(len(df_angles))]
        period = autocorrelation(df_angles["RHip_angle"], df_angles["Time_in_sec"])
        df_angles.drop("Time_in_sec", axis=1, inplace=True)
        
        # Normalize columns
        #df_angles = (df_angles-df_angles.mean())/df_angles.std()
        
        df_metrics = pd.read_csv(metrics_path, sep=";")
        sliced_df_test = df_slice(df_angles, df_metrics, video_name)
        sliced_df_test["Video"] = video_name
        sliced_df_test["Step rate pred"] = (2*60.0)/period
        
        if video_name == test_video_names[0]:
            df_complete_test = sliced_df_test.copy()
        else:
            df_complete_test = pd.concat([df_complete_test, sliced_df_test])

    selected_cols = output_cols.copy()
    selected_cols.append("Video")
    selected_cols.append("Step rate")
    selected_cols.append("Step rate pred")
    df_output_results = df_complete_test[selected_cols]

    models = [ctl_model, ctr_model, frl_model, frr_model]
    
    pd.options.mode.chained_assignment = None
    
    for output_col, model_i in zip(output_cols, models):
        df_output_results_test = predict_videos(df_complete_test, df_output_results, output_col, model_i)
    
    pd.options.mode.chained_assignment = 'warn'
    
    df_output_results_test = df_output_results_test.groupby("Video").mean()
    print("\n\nResults in test videos:")
    print(df_output_results_test)

    
    df_output_results_train = df_complete[selected_cols]
    
    pd.options.mode.chained_assignment = None
    
    for output_col, model_i in zip(output_cols, models):
        df_output_results_train = predict_videos(df_complete, df_output_results_train, output_col, model_i)
    
    pd.options.mode.chained_assignment = 'warn'
    
    df_output_results_train = df_output_results_train.groupby("Video").mean()
    print("\n\nResults in train videos:")
    print(df_output_results_train)


if __name__ == '__main__':
    main()