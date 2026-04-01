import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_process.file_util_pkg import load_npy
from data_visual.plot_pkg import plot_violins
from metrics_pkg import q
import numpy as np
import pandas as pd

session_data = {
    "exp_id":"exp1",
    "patient_id":"p1",
    "subject_id":"ying2",
    "sbmvmt_num":2,
    "num_rep":4,
    "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
}
subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')

val_samples = load_npy(f"{subject_path}/repro/val_processed.npy")
val_samples_df = pd.DataFrame(val_samples)
avg_norm_error = np.array(val_samples_df['avg_norm_error'].values.tolist())
norm_diff_tau = np.array(val_samples_df['norm_diff_tau'].values.tolist())
coverage = np.array(val_samples_df['coverage'].values.tolist())

test_samples = load_npy(f"{subject_path}/repro/test_processed.npy")
test_samples_df = pd.DataFrame(test_samples)
avg_norm_error_test = np.array(test_samples_df['avg_norm_error'].values.tolist())
norm_diff_tau_test = np.array(test_samples_df['norm_diff_tau'].values.tolist())
coverage_test = np.array(test_samples_df['coverage'].values.tolist())

plot_violins(
    title=r"Normalized Metric Dist.",
    data_list=[[avg_norm_error.flatten(),avg_norm_error_test.flatten(),norm_diff_tau.flatten(),norm_diff_tau_test.flatten()],
               [np.mean(avg_norm_error,axis=1),np.mean(avg_norm_error_test,axis=1),np.mean(norm_diff_tau,axis=1),np.mean(norm_diff_tau_test,axis=1)]],
    axis_num = 2,
    x_labels=[r"Average Norm $\epsilon$ (Seen)",r"Average Norm $\epsilon$ (Unseen)",r"Norm $\Delta\tau_{peak}$ (Seen)",r"Norm $\Delta\tau_{peak}$ (Unseen)"],
    axis_title=["All Joints","Mean Joints"])

plot_violins(
    title="Coverage Dist",
    data_list=[[coverage.flatten(),coverage_test.flatten()],
               [np.mean(coverage,axis=1),np.mean(coverage_test,axis=1)]],
    axis_num = 2,
    x_labels=[r"Coverage % (Seen)",r"Coverage % (Unseen)"],
    axis_title=["All Joints","Mean Joints"])

# data_list_per_joint = []
# coverage_per_joint = []
# x_label_list = []
# for i,joint in enumerate(q):
#     data_list_per_joint.append([avg_norm_error[:,i],avg_norm_error_test[:,i],norm_diff_tau[:,i],norm_diff_tau_test[:,i]])
#     coverage_per_joint.append([coverage[:,i],coverage_test[:,i]])
#     x_label_list.append(f"{joint}")
    
# plot_violins(
#     data_list=data_list_per_joint,
#     title="Joint Normalized Metric Dist.",
#     axis_num = len(q),
#     x_labels=[r"Average Norm $\epsilon$ (Seen)",r"Average Norm $\epsilon$ (Unseen)",r"Norm $\Delta\tau_{peak}$ (Seen)",r"Norm $\Delta\tau_{peak}$ (Unseen)"],
#     axis_title=x_label_list)
# plot_violins(
#     data_list=coverage_per_joint,
#     title="Joint Coverage Dist.",
#     axis_num = len(q),
#     x_labels=[r"Coverage % (Seen)",r"Coverage % (Unseen)"],
#     axis_title=x_label_list)

