import sys,os
import numpy as np
import time as times

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_process.csv_pkg import compile_train_val_test_data

from tpgmm_pkg.TPGMM import TPGMM
from tpgmm_pkg.tpgmm_util import arrange_data,get_optim_nbGauss#,save_results
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True,precision=4) # suppress scientific notation

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from data_process.plot_pkg import compare_multi_dim_data,plot_mean_ci,interactive_plot,split_plot_all

session_data = {
    "subject_id":"exp1/p1/ying",
    "task_id":"task_1",
    "sbmvmt_num":2,
    "num_rep":4,
    "variants":["var_1","var_2","var_3","var_4","var_5","var_6"]
}
subject_path = os.path.join(os.path.dirname(__file__), '../..',f'logs/pycorc_recordings/{session_data["subject_id"]}/{session_data["task_id"]}')

train_list,val_list,test_list = compile_train_val_test_data(session_data,subject_path,False)
all_data = []
for train in train_list:
    all_data.append(train["data"])
all_data = np.vstack(all_data)

start = times.perf_counter()
num_of_gauss = get_optim_nbGauss(all_data)
end = times.perf_counter()
print(f"BIC Search:Elapsed \t= {(end - start):.4f}s\n")

"""
transform training and test data
"""
num_of_frames = session_data["sbmvmt_num"]+1
num_of_dims = 30
train_tp_data,train_sample_list = arrange_data(train_list,num_of_frames,num_of_dims,num_of_gauss)
_,val_sample_list = arrange_data(val_list,num_of_frames,num_of_dims,num_of_gauss)
_,test_sample_list = arrange_data(test_list,num_of_frames,num_of_dims,num_of_gauss)

"""
initialize and fit the GMM 
"""  
start = times.perf_counter()
tpgmm = TPGMM(num_of_gauss,num_of_frames,num_of_dims,priors=np.ones((num_of_gauss,)),kP=0,kV=0,diagRegFact=1e-8,version="fast")
tpgmm.init_gmm(train_tp_data)
LL = tpgmm.fit_em(nbMinSteps=3,nbMaxSteps=100,maxDiffLL=1e-5,updateComp=np.ones((3,1)),tp_data=train_tp_data)
end = times.perf_counter()
compare_multi_dim_data(
    x_list=[np.array([i for i in range(LL.shape[0])])],
    data_list=[LL],
    dim=1,
    labels=["Log-Likelihood"],
    xtype="iter",
    datatype="loglikelihood",
    fig_label=f"Log-Likelihood Convergence",
)
print(f"Fast Training:Elapsed \t= {(end - start):.4f}s")
print() 

"""
Deployment pipeline (condition then GMR)
"""
lastInputIndex = 19  # always arrange data such that [input indices, output indices]
lastOutputIndex = 29
start = times.perf_counter()

"""perform reconstruction for training task parameters"""
recon_list = []
gt_list = []
label_list = []
time_list = []
var_id_list = []
for train_dict,sample in zip(train_list,train_sample_list): 
    DataIn  = sample.Data[:,0:lastInputIndex+1]
    DataOut = sample.Data[:,lastInputIndex+1:]
    expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)
    sample_id = train_dict["id"].split(".")
    var_id_list.append(sample_id[1])
    # save_results(test_dict["time"],DataOut,expected_data,f"{subject_path}/{sample_id[1]}/processed/repro/UBORepro{sample_id[2]}Log.csv")

    time_list.append(train_dict["time"])
    recon_list.append(expected_data)
    gt_list.append(DataOut)
    label_list.append(train_dict["id"])
gt_list_sep_var,unique_var_id = split_plot_all(var_id_list,time_list,gt_list,label_list,rep_split=3,fig_label="Train GT")
recon_list_sep_var,_ = split_plot_all(var_id_list,time_list,recon_list,label_list,rep_split=3,fig_label="Train Recon")

# plot recon & gt's mean and ci for each var
stats_fig,stats_ax = compare_multi_dim_data(
    x_list=[],
    data_list=[],
    dim=10,
    labels=[],
    xtype="time",
    datatype="tau_stats",
    fig_label=f"Train Stats Compare",
    split=3
)
plot_mean_ci(
    time_list[0],
    gt_list_sep_var+recon_list_sep_var,
    fig=stats_fig,
    axs=stats_ax,
    labels=[f"GT {x}" for x in unique_var_id]+[f"Recon {x}" for x in unique_var_id],
    relimit=False,
    split=3
)
interactive_plot(stats_fig,stats_ax)

"""perform reconstruction for validation task parameters"""
recon_list = []
gt_list = []
label_list = []
time_list = []
var_id_list = []
for val_dict,sample in zip(val_list,val_sample_list): 
    DataIn  = sample.Data[:,0:lastInputIndex+1]
    DataOut = sample.Data[:,lastInputIndex+1:]
    expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)
    sample_id = val_dict["id"].split(".")
    var_id_list.append(sample_id[1])
    # save_results(test_dict["time"],DataOut,expected_data,f"{subject_path}/{sample_id[1]}/processed/repro/UBORepro{sample_id[2]}Log.csv")

    time_list.append(val_dict["time"])
    recon_list.append(expected_data)
    gt_list.append(DataOut)
    label_list.append(val_dict["id"])
gt_list_sep_var,unique_var_id = split_plot_all(var_id_list,time_list,gt_list,label_list,rep_split=1,fig_label="Val GT")
recon_list_sep_var,_ = split_plot_all(var_id_list,time_list,recon_list,label_list,rep_split=1,fig_label="Val Recon")

# plot recon & gt's mean and ci for each var
stats_fig,stats_ax = compare_multi_dim_data(
    x_list=[],
    data_list=[],
    dim=10,
    labels=[],
    xtype="time",
    datatype="tau_stats",
    fig_label=f"Val Stats Compare",
    split=1
)
plot_mean_ci(
    time_list[0],
    gt_list_sep_var+recon_list_sep_var,
    fig=stats_fig,
    axs=stats_ax,
    labels=[f"GT {x}" for x in unique_var_id]+[f"Recon {x}" for x in unique_var_id],
    relimit=False,
    split=1
)
interactive_plot(stats_fig,stats_ax)

"""perform reconstruction for test task parameters"""
recon_list = []
gt_list = []
label_list = []
time_list = []
var_id_list = []
for test_dict,sample in zip(test_list,test_sample_list): 
    DataIn  = sample.Data[:,0:lastInputIndex+1]
    DataOut = sample.Data[:,lastInputIndex+1:]
    expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)
    sample_id = test_dict["id"].split(".")
    var_id_list.append(sample_id[1])
    # save_results(test_dict["time"],DataOut,expected_data,f"{subject_path}/{sample_id[1]}/processed/repro/UBORepro{sample_id[2]}Log.csv")

    time_list.append(test_dict["time"])
    recon_list.append(expected_data)
    gt_list.append(DataOut)
    label_list.append(test_dict["id"])
gt_list_sep_var,unique_var_id = split_plot_all(var_id_list,time_list,gt_list,label_list,rep_split=4,fig_label="Test GT")
recon_list_sep_var,_ = split_plot_all(var_id_list,time_list,recon_list,label_list,rep_split=4,fig_label="Test Recon")

# plot recon & gt's mean and ci for each var
stats_fig,stats_ax = compare_multi_dim_data(
    x_list=[],
    data_list=[],
    dim=10,
    labels=[],
    xtype="time",
    datatype="tau_stats",
    fig_label=f"Test Stats Compare",
    split=4
)
plot_mean_ci(
    time_list[0],
    gt_list_sep_var+recon_list_sep_var,
    fig=stats_fig,
    axs=stats_ax,
    labels=[f"GT {x}" for x in unique_var_id]+[f"Recon {x}" for x in unique_var_id],
    relimit=False,
    split=4
)
interactive_plot(stats_fig,stats_ax)

""" END OF PIPELINE"""
end = times.perf_counter()
print(f"Fast Deploy:Elapsed \t= {(end - start):.4f}s")

plt.show(block=True)

