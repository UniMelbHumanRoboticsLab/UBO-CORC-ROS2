#%% 
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

from plot_pkg import compare_multi_dim_data,plot_3d_trajectory,plot_3d_points,split_plot_all
from post_process_pkg import lpf,calc_fixed_diff,calc_mag,segment_sbmvmts,rescale,separate_train_test,separate_train_val,create_dir,split_reps
from csv_pkg import get_raw_data

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyCORC.pycorc_io.xsens.ub_pckg.ub import ub
from pyCORC.pycorc_io.package_utils.unpack_json import get_subject_params

#%% 
""" init session parameters and perform train / test / validation spliting"""
session_data = {
    "subject_id":"exp1/p1/ying",
    "task_id":"task_1",
    "sbmvmt_num":2,
    "num_rep":4,
    "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
}
subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["subject_id"]}')
# separate variants for train test and save splits to csv
case_id,train_var,test_var = separate_train_test(session_data["variants"],f'{subject_path}/{session_data["task_id"]}/train_test_split.csv')
train_set , val_set = separate_train_val(train_var,session_data["num_rep"],f'{subject_path}/{session_data["task_id"]}/train_val_split.csv')

""" init xsens skeleton model """
body_params_rbt,ft_grav,ft_install = get_subject_params(subject_path)
skeleton = ub(body_params_rbt,model="ubo",arm_side="right")
NUM_RFT = 3
rft_keys = ["clav","ua","fa"]

#%% 
""" compile all data across all variations-takes in current p1 actor - task """
""" remove gravity effects from corc data """
""" filter everything with low pass filter """
""" get the external torque generated in that submovement """
""" segment the submovements for each repetition """
""" extract task parameters from each submovement """
full_data_dict = {}
time_list,q_traj_list,qdot_traj_list,hand_3d_traj_list,sbmvmt_list,rep_label_list = [],[],[],[],[],[]
fc = 10  # cut-off frequency
fs = 100
dt = 1/fs
for var in session_data["variants"]:
    full_data_dict[var] = {}
    for rep in range(1,session_data["num_rep"]+1):
        # read the csv file for the current rep
        data_path = f'{subject_path}/{session_data["task_id"]}/{var}/raw/UBORecord{rep}Log.csv'
        # create the following subdirectories if needed
        create_dir(f'{subject_path}/{session_data["task_id"]}/{var}/processed/tp')
        create_dir(f'{subject_path}/{session_data["task_id"]}/{var}/processed/index')
        print(data_path)

        full_data,time_data_unscaled,corc_data_unscaled,q_traj_unscaled = get_raw_data(data_path)
        time_data_unscaled = time_data_unscaled - time_data_unscaled[0]

        # add the initial installation bias, remove average bias collected during Take_bias, remove gravity bias using xsens data FROM corc_data
        robot_ees = skeleton.fkine(q_traj_unscaled.tolist())
        offset2 = []
        total_weight_traj = []
        for i in range(NUM_RFT):
            # get initial_bias
            offset2 += ft_install[rft_keys[i]]
            
            # calculate traj of current sensor weight in sensor frame
            pose_traj = robot_ees[i + 1]
            weight_comp = np.array([x * ft_grav[rft_keys[i]] for x in [0,0,-1]])
            weight_traj = np.einsum('nji,j->ni', pose_traj.R, weight_comp) # transpose + matmul
            weight_traj = np.hstack([weight_traj, np.zeros((weight_traj.shape[0], 3))])  # append nx3 zeros
            total_weight_traj.append(weight_traj)
        total_weight_traj = np.hstack(total_weight_traj) 
        offset2 = np.array(offset2)
        corc_data_unscaled = corc_data_unscaled + np.array(offset2) - total_weight_traj

        """ Filter """
        # filter the original data
        plot_results = False # check if filter fucks up the data
        dt_unscaled = np.mean(np.diff(time_data_unscaled))
        corc_data   = lpf(time_data_unscaled,corc_data_unscaled,ts=dt_unscaled,fc=fc,datatype=f"wrenches_{var}-Rep{rep}",   plot_results=plot_results)
        q_traj      = lpf(time_data_unscaled,q_traj_unscaled,   ts=dt_unscaled,fc=fc,datatype=f"q_{var}-Rep{rep}",          plot_results=plot_results)
        # calculate joint space kinematic and filter
        qdot_traj   = calc_fixed_diff(q_traj, dt=dt_unscaled)
        qdot_traj   = lpf(time_data_unscaled, qdot_traj, ts=dt_unscaled, fc=1.5, datatype=f"qdot_{var}-Rep{rep}", plot_results=plot_results)
        # calculate and filter task space kinematic data for active movement segmentation
        hand_pos_traj   = skeleton.fkine(q_traj.tolist())[0].t
        hand_vel        = calc_fixed_diff(hand_pos_traj, dt=dt_unscaled)
        hand_vel        = lpf(time_data_unscaled,hand_vel,ts=dt_unscaled,fc=1.5,datatype=f"hand_vel_{var}-Rep{rep}",plot_results=plot_results)
        hand_speed      = calc_mag(hand_vel)
        # calculate and filter joint space torques generated by each RFT and total
        taus_traj       = skeleton.get_joints_torques_traj(q_traj, corc_data)
        for rft_key in rft_keys:
            taus_traj[rft_key]["filtered"] = lpf(time_data_unscaled,taus_traj[rft_key]["raw"],ts=dt_unscaled,fc=fc,datatype=f"tau_{rft_key}_{var}-Rep{rep}",plot_results=plot_results)
        taus_traj["total"]["filtered"] = lpf(time_data_unscaled,taus_traj["total"]["raw"],ts=dt_unscaled,fc=fc,datatype=f"tau_total_{var}-Rep{rep}",plot_results=plot_results)
        
        """ Downsample """
        # downsample for easier model learning
        plot_results = False # check if downsample fucks up the data
        time_data       = np.arange(time_data_unscaled[0],time_data_unscaled[-1]+dt_unscaled,1/fs,dtype=float)
        corc_data       = rescale(time_data_unscaled, corc_data,        time_data,  datatype=f"wrenches_{var}-Rep{rep}",      plot_results=plot_results)
        q_traj          = rescale(time_data_unscaled, q_traj,           time_data,  datatype=f"q_{var}-Rep{rep}",             plot_results=plot_results)
        qdot_traj       = rescale(time_data_unscaled, qdot_traj,        time_data,  datatype=f"qdot_{var}-Rep{rep}",          plot_results=plot_results)
        hand_pos_traj   = rescale(time_data_unscaled, hand_pos_traj,    time_data,  datatype=f"hand_pos_{var}-Rep{rep}",      plot_results=plot_results)
        hand_speed      = rescale(time_data_unscaled, hand_speed,       time_data,  datatype=f"hand_speed_{var}-Rep{rep}",    plot_results=plot_results)
        for rft_key in rft_keys+["total"]:
            taus_traj[rft_key]["filtered-rescaled"] = rescale(time_data_unscaled, taus_traj[rft_key]["filtered"], time_data,  datatype=f"tau_{rft_key}_{var}-Rep{rep}",      plot_results=plot_results)

        """ Segment """
        # index active movement
        _,start_end_indices = segment_sbmvmts(time_data,hand_pos_traj,hand_speed,1,data_path=f'{subject_path}/{session_data["task_id"]}/{var}/processed/index/UBOStartEnd{rep}.txt',redo=False)
        time_data       = time_data[start_end_indices[0]:start_end_indices[-1]]-time_data[start_end_indices[0]]
        corc_data       = corc_data[start_end_indices[0]:start_end_indices[-1],:]
        q_traj          = q_traj[start_end_indices[0]:start_end_indices[-1],:10] # from here just collect the first 10 joints
        qdot_traj       = qdot_traj[start_end_indices[0]:start_end_indices[-1],:10]
        hand_pos_traj   = hand_pos_traj[start_end_indices[0]:start_end_indices[-1],:]
        hand_speed      = hand_speed[start_end_indices[0]:start_end_indices[-1],:]
        for rft_key in rft_keys+["total"]:
            taus_traj[rft_key]["filtered-rescaled"] = taus_traj[rft_key]["filtered-rescaled"][start_end_indices[0]:start_end_indices[-1],:]

        # use first demo as the reference for time alignment
        if var == session_data["variants"][0] and rep == 1:
            time_data_norm  = time_data / time_data[-1]
            time_data       = time_data_norm
            num_samples     = time_data.shape[0]
        else:
            time_data       = time_data / time_data[-1]
            time_data_norm  = np.linspace(0, 1, num_samples)

        """ Time Normalization """
        # rescale all data to the reference demo time
        plot_results = False # check if time normalization fucks up the data
        corc_data       = rescale(time_data, corc_data,     time_data_norm, datatype=f"wrenches_norm_{var}-Rep{rep}",   plot_results=plot_results)
        q_traj          = rescale(time_data, q_traj,        time_data_norm, datatype=f"q_norm_{var}-Rep{rep}",          plot_results=plot_results)
        qdot_traj       = rescale(time_data, qdot_traj,     time_data_norm, datatype=f"qdot_norm_{var}-Rep{rep}",       plot_results=plot_results)
        q_traj_rad      = np.deg2rad(q_traj)
        qdot_traj_rad   = np.deg2rad(qdot_traj)


        hand_pos_traj   = rescale(time_data, hand_pos_traj, time_data_norm, datatype=f"hand_pos_norm_{var}-Rep{rep}",   plot_results=plot_results)
        hand_speed      = rescale(time_data, hand_speed,    time_data_norm, datatype=f"hand_speed_norm_{var}-Rep{rep}", plot_results=plot_results)
        for rft_key in rft_keys+["total"]:
            taus_traj[rft_key]["filtered-rescaled"] = rescale(time_data, taus_traj[rft_key]["filtered-rescaled"], time_data_norm, datatype=f"tau_norm_{rft_key}_{var}-Rep{rep}",    plot_results=plot_results)

        """ Segment """
        # segment rescaled submovements
        indices_traj,sbmvmt_indices = segment_sbmvmts(time_data_norm,hand_pos_traj,hand_speed,session_data["sbmvmt_num"],data_path=f'{subject_path}/{session_data["task_id"]}/{var}/processed/index/UBOIndex{rep}.txt',redo=False)

        """ Extract TP """
        # extract task parameters from each submovements (Input: joint kinematics, output: joint torques)
        q = ['trunk_ie','trunk_aa','trunk_fe',
                'clav_dep_ev','clav_prot_ret',
                'shoulder_fe','shoulder_aa','shoulder_ie',
                'elbow_fe','elbow_ps']
        qdot = [f"{joint}_dot" for joint in q]
        tau_list = [f"tau_{joint}" for joint in q]
        num_dim = len(q)+len(qdot)+len(tau_list)
        task_params_A   = np.vstack([np.eye(num_dim).flatten() for _ in range(sbmvmt_indices.shape[0])])
        task_params_b   = np.hstack([q_traj_rad[sbmvmt_indices,:], qdot_traj_rad[sbmvmt_indices,:],np.zeros((sbmvmt_indices.shape[0],len(tau_list)))])  # tau should have zero offset
        task_params     = np.hstack([task_params_A, task_params_b])
        tp_df = pd.DataFrame(task_params, columns=[f"A{j}" for j in range(1,num_dim*num_dim+1)] + [f"b{j}" for j in range(1,num_dim+1)])
        tp_df.to_csv(f'{subject_path}/{session_data["task_id"]}/{var}/processed/tp/UBOTP{rep}Log.csv',index=False)

        """ Save Post Processed Data """
        # compile all data
        full_data_dict[var][rep] = {
            "time": time_data_norm,
            "corc": corc_data,
            "q": q_traj,
            "qdot": qdot_traj,
            "taus": taus_traj,
            "sbmvmt": sbmvmt_indices,
            "task_params": task_params_b
        }
        full_processed_data = np.hstack([time_data_norm[:,np.newaxis],indices_traj,q_traj_rad,qdot_traj_rad,taus_traj["total"]["filtered-rescaled"]])
        df_column = ["norm_time","index"]+q+qdot+tau_list
        full_df = pd.DataFrame(full_processed_data,columns=df_column)
        full_df.to_csv(f'{subject_path}/{session_data["task_id"]}/{var}/processed/UBORecord{rep}Log.csv',index=False)

        # for plotting
        time_list.append(time_data_norm)
        q_traj_list.append(q_traj_rad)
        qdot_traj_list.append(qdot_traj_rad)
        hand_3d_traj_list.append(hand_pos_traj)
        sbmvmt_list.append(indices_traj)  
        rep_label_list.append(f'{var}-Rep{rep}')

fig,ax = plot_3d_trajectory(traj_list=hand_3d_traj_list,label_list=rep_label_list)
split_plot_all(session_data["variants"],time_list,q_traj_list,rep_label_list,rep_split=session_data["num_rep"],data_type="q_rad",fig_label="q_rad data")

# compare_multi_dim_data(
#         time_list,
#         qdot_traj_list,
#         10,
#         rep_label_list,
#         'Time(s)',
#         "qdot_rad",
#         sharex=True,
#         semilogx=False,
#         fig_label=f"joint angle velocity - rad/s",
#         show_stats=True)

# compare_multi_dim_data(
#         time_list,
#         sbmvmt_list,
#         1,
#         rep_label_list,
#         'Time(s)',
#         "sbmvmt",
#         sharex=True,
#         semilogx=False,
#         fig_label="submovement")

# sanity check velocity (zero crossing should match peak positions)
# for i,(qtraj,qdot_traj) in enumerate(zip(q_traj_list,qdot_traj_list)):
#     compare_multi_dim_data(
#             [time_data_norm,time_data_norm],
#             [qtraj,qdot_traj],
#             10,
#             ["traj","vel"],
#             'Time(s)',
#             "qqdot",
#             sharex=True,
#             semilogx=False,
#             fig_label=f"qqdot_{i}",
#             show_zero_cross=True)

#%% 
""" check individual RFT wrench and generated torques for all vars-reps """
for i, rft_key in enumerate(rft_keys+["total"]):
    time_list,force_list,moment_list,wrench_list,taus_list,rep_label_list = [],[],[],[],[],[]

    for var in session_data["variants"]:
        for rep in range(1,session_data["num_rep"]+1):
            time_data = full_data_dict[var][rep]["time"]
            corc_data = full_data_dict[var][rep]["corc"]
            tau_data = full_data_dict[var][rep]["taus"][rft_key]["filtered-rescaled"]

            if rft_key != "total":
                wrench = corc_data[:,i*6:i*6+6]
                force_list.append(wrench[:,0:3])
                moment_list.append(wrench[:,3:])
                wrench_list.append(wrench)
            time_list.append(time_data)
            taus_list.append(tau_data)
            rep_label_list.append(f'{var}-Rep{rep}')

    if rft_key != "total":
        # compare_multi_dim_data(
        #     time_list,
        #     wrench_list,
        #     6,
        #     rep_label_list,
        #     'Time(s)',
        #     f"{rft_key}",
        #     sharex=True,
        #     semilogx=False,
        #     fig_label=f"{rft_key}")
        continue
    else:
        split_plot_all(session_data["variants"],time_list,taus_list,rep_label_list,rep_split=session_data["num_rep"],data_type="tau",fig_label="tau data")

plt.show()
