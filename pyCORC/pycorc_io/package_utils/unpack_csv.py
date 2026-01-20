import pandas as pd
import os

def get_raw_data(data_path):
    
    
    data = pd.read_csv(data_path)
    time_data = data["elapsed_time"].values
    corc_data = data[["F1x", "F1y", "F1z", "T1x", "T1y", "T1z",
            "F2x", "F2y", "F2z", "T2x", "T2y", "T2z",
            "F3x", "F3y", "F3z", "T3x", "T3y", "T3z"]].values
    joints = ['trunk_ie','trunk_aa','trunk_fe',
                'scapula_de','scapula_pr',
                'shoulder_fe','shoulder_aa','shoulder_ie',
                'elbow_fe','elbow_ps',
                'wrist_fe','wrist_dev']
    right_xsens_data = data[[f"{joint}_right" for joint in joints]].values

    return data,time_data,corc_data,right_xsens_data

def save_file(path,df:pd.DataFrame):

    df.to_csv(path,index=True)