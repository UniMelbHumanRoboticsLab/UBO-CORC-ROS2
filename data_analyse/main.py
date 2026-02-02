import sys,os
import matplotlib.pyplot as plt
import numpy as np

from metrics_pkg import compute_impulse,compute_peak_tau

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_process.csv_pkg import compile_val_test_repro

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_process.plot_pkg import compare_multi_dim_data,plot_mean_ci

session_data = {
    "subject_id":"exp1/p1/vincent",
    "task_id":"task_1",
    "sbmvmt_num":4,
    "num_rep":4,
    "variants":["var_1","var_2","var_3","var_4","var_5","var_6"]
}

subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["subject_id"]}/{session_data["task_id"]}')

val_list,test_list = compile_val_test_repro(session_data,subject_path)



for val_dict in val_list:
    val_dict["impulse_gt"] = compute_impulse(val_dict["time"],val_dict["gt"])
    val_dict["peak_tau_gt"] = compute_peak_tau(val_dict["gt"])

    val_dict["impulse_repro"] = compute_impulse(val_dict["time"],np.abs(val_dict["repro"]))
    val_dict["peak_tau_repro"] = compute_peak_tau(val_dict["repro"])
    
    compare_multi_dim_data(
        x_list=[val_dict["time"],val_dict["time"]],
        data_list=[val_dict["gt"],np.abs(val_dict["gt"])],
        dim=10,
        labels=["gt","gt_abs"],
        xtype="time",
        datatype="tau_gt",
        fig_label=f"GT_Repro",
        show_stats=True
    )
    plt.show(block=True)

p = 0
