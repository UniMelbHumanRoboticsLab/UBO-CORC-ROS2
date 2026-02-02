import sys,os
import numpy as np
import time as times

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_process.csv_pkg import compile_train_val_test_data

from tpgmm_pkg.TPGMM import *
from tpgmm_pkg.tpgmm_util import arrange_data,get_optim_nbGauss,save_results
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True,precision=4) # suppress scientific notation

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from data_process.plot_pkg import compare_multi_dim_data,plot_mean_ci

session_data = {
    "subject_id":"exp1/p1/vincent",
    "task_id":"task_1",
    "sbmvmt_num":4,
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
num_of_gauss = 4#get_optim_nbGauss(all_data)
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

# import time as times
# start = times.perf_counter()
# tpgmm_slow = TPGMM(num_of_gauss,num_of_frames,num_of_dims,priors=np.zeros((num_of_gauss,1)),kP=0,kV=0,diagRegFact=1e-8,version="slow")
# tpgmm_slow.init_gmm(train_tp_data)
# tpgmm_slow.fit_em(nbMinSteps=5,nbMaxSteps=100,maxDiffLL=0.01,updateComp=np.ones((3,1)),tp_data=train_tp_data)
# end = times.perf_counter()
# print(f"Slow Training:Elapsed \t= {(end - start):.4f}s")

# print() 
# print("fast-old diff Mu:", np.max(np.abs(tpgmm_slow.Mu - tpgmm.Mu)),tpgmm.Mu.shape)
# print("fast-old diff Sigma:", np.max(np.abs(tpgmm_slow.Sigma - tpgmm.Sigma)),tpgmm.Sigma.shape)
# print("fast-old diff Priors:", np.max(np.abs(tpgmm_slow.priors - tpgmm.priors)))
# print("fast-old diff Pix:", np.max(np.abs(tpgmm_slow.Pix - tpgmm.Pix)))
# print()

"""
Deployment pipeline (condition then GMR)
"""
lastInputIndex = 19  # always arrange data such that [input indices, output indices]
lastOutputIndex = 29
start = times.perf_counter()

# perform reconstruction for validation task parameters
recon_list = []
gt_list = []
labels = []
time_list = []
for val_dict,sample in zip(val_list,val_sample_list): 
    DataIn  = sample.Data[:,0:lastInputIndex+1]
    DataOut = sample.Data[:,lastInputIndex+1:]
    expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)

    sample_id = val_dict["id"].split(".")
    save_results(val_dict["time"],DataOut,expected_data,f"{subject_path}/{sample_id[1]}/processed/repro/UBORepro{sample_id[2]}Log.csv")

# perform reconstruction for test task parameters
for test_dict,sample in zip(test_list,test_sample_list): 
    DataIn  = sample.Data[:,0:lastInputIndex+1]
    DataOut = sample.Data[:,lastInputIndex+1:]
    expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)
    sample_id = test_dict["id"].split(".")
    save_results(test_dict["time"],DataOut,expected_data,f"{subject_path}/{sample_id[1]}/processed/repro/UBORepro{sample_id[2]}Log.csv")

    time_list.append(test_dict["time"])
    recon_list.append(expected_data)
    gt_list.append(DataOut)
    labels.append(test_dict["id"])

compare_multi_dim_data(
    x_list=time_list,
    data_list=recon_list,
    dim=10,
    labels=labels,
    xtype="time",
    datatype="tau_recon",
    fig_label=f"Test Reconstruction",
    show_stats=True
)
compare_multi_dim_data(
    x_list=time_list,
    data_list=gt_list,
    dim=10,
    labels=labels,
    xtype="time",
    datatype="tau_gt",
    fig_label=f"Test GT",
    show_stats=True
)

stats_fig,stats_ax = compare_multi_dim_data(
    x_list=[],
    data_list=[],
    dim=10,
    labels=[],
    xtype="time",
    datatype="tau_stats",
    fig_label=f"Test Stats Compare"
)
plot_mean_ci(
    time_list[0],
    [gt_list,recon_list],
    fig=stats_fig,
    axs=stats_ax,
    labels=["Ground Truth","Reconstruction"],
    relimit=True
)

end = times.perf_counter()
print(f"Fast Deploy:Elapsed \t= {(end - start):.4f}s")

plt.show(block=True)

