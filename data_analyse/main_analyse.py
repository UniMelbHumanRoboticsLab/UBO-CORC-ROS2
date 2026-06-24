import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_process.file_util_pkg import load_npy
from data_visual.plot_pkg import plot_violins
from metrics_pkg import q
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_subject = True
plot_per_patient = False
if all_subject:
    """
    Every subject
    """
    all_patients_error = []
    all_patients_coverage = []
    all_patients_gaussian = []
    for p in range(1,4):
        if p == 1:
            sm_num = 2
        else:
            sm_num=4
        
        all_val_avg_norm_error = []
        all_val_norm_diff_tau = []
        all_val_coverage = []
        
        all_test_avg_norm_error = []
        all_test_norm_diff_tau = []
        all_test_coverage = []
        
        all_metric_per_subject = []
        all_coverage_per_subject = []
        
        num_of_gaussians = []
        for sub in range(11,25):
            session_data = {
                "exp_id":"exp1_trained2",
                "patient_id":f"p{p}",
                "subject_id":f"sub{sub}",
                "sbmvmt_num":sm_num,
                "num_rep":4,
                "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
            }
            subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
            
            val_samples = load_npy(f"{subject_path}/repro/val_processed.npy")
            val_samples_df = pd.DataFrame(val_samples)
      
            test_samples = load_npy(f"{subject_path}/repro/test_processed.npy")
            test_samples_df = pd.DataFrame(test_samples)
            
            val_avg_norm_error = val_samples_df['avg_norm_error'].tolist()
            val_norm_diff_tau = val_samples_df['norm_diff_tau'].tolist()
            val_coverage = val_samples_df['coverage'].tolist()
            
            test_avg_norm_error = test_samples_df['avg_norm_error'].tolist()
            test_norm_diff_tau = test_samples_df['norm_diff_tau'].tolist()
            test_coverage = test_samples_df['coverage'].tolist()
            
            all_val_avg_norm_error.append(val_avg_norm_error)
            all_val_norm_diff_tau.append(val_norm_diff_tau)
            all_val_coverage.append(val_coverage)
            
            all_test_avg_norm_error.append(test_avg_norm_error)
            all_test_norm_diff_tau.append(test_norm_diff_tau)
            all_test_coverage.append(test_coverage)
            
            num_of_gaussians.append(val_samples_df['tpgmm_param'].values[0::4].tolist())
            all_metric_per_subject.append([np.array(val_avg_norm_error) ,np.array(test_avg_norm_error) ,np.array(val_norm_diff_tau) ,np.array(test_norm_diff_tau) ])
            all_coverage_per_subject.append([np.array(val_coverage),np.array(test_coverage)])
        
        """plot all subject as a spread"""
        if plot_per_patient:
            plot_violins(
                data_list=all_metric_per_subject,
                title=f"p{p} Normalized Metric Spread.",
                axis_num = 14,
                x_labels=[r"Average Norm $\epsilon$"+"\n(Seen)",r"Average Norm $\epsilon$"+"\n(Unseen)",r"Norm $\Delta\tau_{peak}$"+"\n(Seen)",r"Norm $\Delta\tau_{peak}$"+"\n(Unseen)"],
                axis_title=[f"sub{x}" for x in range(11,25)])
            plot_violins(
                data_list=all_coverage_per_subject,
                title=f"p{p} Coverage Spread.",
                axis_num = 14,
                x_labels=[r"Coverage %"+"\n(Seen)",r"Coverage %"+"\n(Unseen)"],
                axis_title=[f"sub{x}" for x in range(11,25)])
        
        all_val_avg_norm_error = np.vstack(all_val_avg_norm_error)
        all_val_norm_diff_tau = np.vstack(all_val_norm_diff_tau)
        all_val_coverage = np.vstack(all_val_coverage)
        
        all_test_avg_norm_error = np.vstack(all_test_avg_norm_error)
        all_test_norm_diff_tau = np.vstack(all_test_norm_diff_tau)
        all_test_coverage = np.vstack(all_test_coverage)
        
        all_error = [all_val_avg_norm_error,all_test_avg_norm_error,all_val_norm_diff_tau,all_test_norm_diff_tau]
        all_coverage = [all_val_coverage,all_test_coverage ]
        
        
        """plot all subject as a whole"""
        if plot_per_patient:
            plot_violins(
                title=fr"p{p} Normalized Metric Dist.",
                data_list=[all_error],
                axis_num = 1,
                x_labels=[r"Average Norm $\epsilon$"+"\n(Seen)",r"Average Norm $\epsilon$"+"\n(Unseen)",r"Norm $\Delta\tau_{peak}$"+"\n(Seen)",r"Norm $\Delta\tau_{peak}$"+"\n(Unseen)"],
                axis_title=["All Joints"])
            plot_violins(
                title=fr"p{p} Coverage Dist",
                data_list=[all_coverage],
                axis_num = 1,
                x_labels=[r"Coverage %"+"\n(Seen)",r"Coverage %"+"\n(Unseen)"],
                axis_title=["All Joints"])
        
        """"plot all joints as a spread"""
        if plot_per_patient:
            data_list_per_joint = []
            coverage_per_joint = []
            axis_title_list = []
            for i,joint in enumerate(q):
                data_list_per_joint.append([all_val_avg_norm_error[:,i],all_test_avg_norm_error[:,i],all_val_norm_diff_tau[:,i],all_test_norm_diff_tau[:,i]])
                coverage_per_joint.append([all_val_coverage[:,i],all_test_coverage[:,i]])
                axis_title_list.append(f"{joint}")
                
            plot_violins(
                data_list=data_list_per_joint,
                title=f"p{p} Joint Normalized Metric Dist",
                axis_num = len(q),
                x_labels=[r"Average Norm $\epsilon$"+"\n(Seen)",r"Average Norm $\epsilon$"+"\n(Unseen)",r"Norm $\Delta\tau_{peak}$"+"\n(Seen)",r"Norm $\Delta\tau_{peak}$"+"\n(Unseen)"],
                axis_title=axis_title_list)
            plot_violins(
                data_list=coverage_per_joint,
                title=f"p{p} Joint Coverage Dist",
                axis_num = len(q),
                x_labels=[r"Coverage %"+"\n(Seen)",r"Coverage %"+"\n(Unseen)"],
                axis_title=axis_title_list)
            plt.show()
        
        all_patients_error.append(all_error)
        all_patients_coverage.append(all_coverage)
        num_of_gaussians = np.hstack(num_of_gaussians)
        all_patients_gaussian.append([num_of_gaussians])

    plot_violins(
        title="Patient Normalized Metric Dist",
        data_list=all_patients_error,
        axis_num = 3,
        x_labels=[r"Average Norm $\epsilon$"+"\n(Seen)",r"Average Norm $\epsilon$"+"\n(Unseen)",r"Norm $\Delta\tau_{peak}$"+"\n(Seen)",r"Norm $\Delta\tau_{peak}$"+"\n(Unseen)"],
        axis_title=["p1","p2","p3"])
    plot_violins(
        title="Patient Coverage Dist",
        data_list=all_patients_coverage,
        axis_num = 3,
        x_labels=[r"Coverage %"+"\n(Seen)",r"Coverage %"+"\n(Unseen)"],
        axis_title=["p1","p2","p3"])
    plot_violins(
        title="Patient Gaussian Kernel Distribution",
        data_list=all_patients_gaussian,
        axis_num = 3,
        x_labels=[r"Num of Gaussians"],
        axis_title=["p1","p2","p3"],
        remove_outlier=False)
    plt.show()
else:
    """
    Single subject
    """
    session_data = {
        "exp_id":"exp1_trained2",
        "patient_id":"p2",
        "subject_id":"sub11",
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
    check = np.mean(avg_norm_error_test,axis=0)
    norm_diff_tau_test = np.array(test_samples_df['norm_diff_tau'].values.tolist())
    coverage_test = np.array(test_samples_df['coverage'].values.tolist())
    
    plot_violins(
        title=r"Normalized Metric Dist.",
        data_list=[[avg_norm_error ,avg_norm_error_test ,norm_diff_tau ,norm_diff_tau_test ],
                   [np.mean(avg_norm_error,axis=1),np.mean(avg_norm_error_test,axis=1),np.mean(norm_diff_tau,axis=1),np.mean(norm_diff_tau_test,axis=1)]],
        axis_num = 2,
        x_labels=[r"Average Norm $\epsilon$ \n(Seen)",r"Average Norm $\epsilon$ \n(Unseen)",r"Norm $\Delta\tau_{peak}$ \n(Seen)",r"Norm $\Delta\tau_{peak}$ (Unseen)"],
        axis_title=["All Joints","Mean Joints"])
    
    plot_violins(
        title="Coverage Dist",
        data_list=[[coverage ,coverage_test ],
                   [np.mean(coverage,axis=1),np.mean(coverage_test,axis=1)]],
        axis_num = 2,
        x_labels=[r"Coverage % \n(Seen)",r"Coverage % \n(Unseen)"],
        axis_title=["All Joints","Mean Joints"])
    
    data_list_per_joint = []
    coverage_per_joint = []
    axis_title_list = []
    for i,joint in enumerate(q):
        data_list_per_joint.append([avg_norm_error[:,i],avg_norm_error_test[:,i],norm_diff_tau[:,i],norm_diff_tau_test[:,i]])
        coverage_per_joint.append([coverage[:,i],coverage_test[:,i]])
        axis_title_list.append(f"{joint}")
        
    plot_violins(
        data_list=data_list_per_joint,
        title="Joint Normalized Metric Dist.",
        axis_num = len(q),
        x_labels=[r"Average Norm $\epsilon$ (Seen)",r"Average Norm $\epsilon$ (Unseen)",r"Norm $\Delta\tau_{peak}$ (Seen)",r"Norm $\Delta\tau_{peak}$ (Unseen)"],
        axis_title=axis_title_list)
    plot_violins(
        data_list=coverage_per_joint,
        title="Joint Coverage Dist.",
        axis_num = len(q),
        x_labels=[r"Coverage % (Seen)",r"Coverage % (Unseen)"],
        axis_title=axis_title_list)
    
    import matplotlib.pyplot as plt
    plt.show()
