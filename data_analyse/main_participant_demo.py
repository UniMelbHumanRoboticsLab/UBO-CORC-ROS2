import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

sys.path.append(os.path.join(os.path.dirname(__file__)))
from stats_pkg import compute_central_tendency
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyCORC.pycorc_io.package_utils.unpack_json import get_subject_file

subjects_dict_list = []
for sub in range(1,25):
    
    session_data = {
        "exp_id":"exp1_trained2",
        "subject_id":f"sub{sub}",
    }
    body_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/subject_measurements/{session_data["subject_id"]}')
    subject_param = get_subject_file(body_path)
    subject_param["sub_id"] = session_data["subject_id"]
    required_keys = ['sub_id','age', 'gender','body_height','p1_interaction','p2_interaction','p3_interaction']
    reduced_subject_param = {k: subject_param[k] for k in required_keys}
    subjects_dict_list.append(reduced_subject_param)
subjects_df = pd.DataFrame(subjects_dict_list)

# Calculate mean and SEM for all numeric columns by gender
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)
result = subjects_df[['age','body_height']].agg(['mean','sem','median', iqr,'min','max'])
print(result)
print()

percentages = subjects_df['gender'].value_counts()
print(percentages)

subjects_df.to_csv("./figures/subject_stats.csv")