import sys,os
import numpy as np
np.set_printoptions(suppress=True,precision=4) # suppress scientific notation
import time as times

from tpgmm_pkg.TPGMM import TPGMM
from tpgmm_pkg.tpgmm_util import arrange_data,get_optim_nbGauss,save_results

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from data_visual.plot_pkg import compare_multi_dim_data,plot_mean_ci,interactive_plot,split_plot_all
from data_process.file_util_pkg import create_dir,compile_train_val_test_data

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
num_of_gauss = 1#get_optim_nbGauss(all_data)
end = times.perf_counter()
print(f"BIC Search:Elapsed \t= {(end - start):.4f}s\n")

"""
Transform training and test data
"""
num_of_frames = session_data["sbmvmt_num"]+1
num_of_dims = 30
train_tp_data,train_sample_list = arrange_data(train_list,num_of_frames,num_of_dims,num_of_gauss)
_,val_sample_list = arrange_data(val_list,num_of_frames,num_of_dims,num_of_gauss)
_,test_sample_list = arrange_data(test_list,num_of_frames,num_of_dims,num_of_gauss)

"""
Initialize and fit the GMM 
"""  
start = times.perf_counter()
tpgmm = TPGMM(num_of_gauss,num_of_frames,num_of_dims,priors=np.ones((num_of_gauss,)),kP=0,kV=0,diagRegFact=1e-8,version="fast")
tpgmm.init_gmm(train_tp_data)
LL = tpgmm.fit_em(nbMinSteps=3,nbMaxSteps=100,maxDiffLL=1e-5,updateComp=np.ones((3,1)),tp_data=train_tp_data)
end = times.perf_counter()
# compare_multi_dim_data(
#     x_list=[np.array([i for i in range(LL.shape[0])])],
#     data_list=[LL],
#     dim=1,
#     labels=["Log-Likelihood"],
#     xtype="iter",
#     datatype="loglikelihood",
#     fig_label=f"Log-Likelihood Convergence",
# )
print(f"Fast Training:Elapsed \t= {(end - start):.4f}s")
print() 

"""
Deployment pipeline (condition then GMR)
"""
lastInputIndex = 19  # always arrange data such that [input indices, output indices]
lastOutputIndex = 29
start = times.perf_counter()

"""
compile training data
"""
gt_list = []
id_list = []
time_list = []
var_id_list = []
for train_dict,sample in zip(train_list,train_sample_list): 
    DataIn  = sample.Data[:,0:lastInputIndex+1]
    DataOut = sample.Data[:,lastInputIndex+1:]
    sample_id = train_dict["id"].split(".")
    var_id_list.append(sample_id[0])

    time_list.append(train_dict["time"])
    gt_list.append(DataOut)
    id_list.append(train_dict["id"])
gt_list_per_var,unique_var_id_list = split_plot_all(var_id_list,time_list,gt_list,id_list,rep_split=3,fig_label="Train GT")

# get the mean of training samples for each variation, repeat once for the comparator as there is only 1 validation sample
train_mean_per_var = []
for samples_per_var in gt_list_per_var:
    mean = np.mean(np.array(samples_per_var),axis=0)
    train_mean_per_var.append(mean)
    
"""
perform reconstruction for validation task parameters
"""
recon_list = []
gt_list = []
id_list = []
time_list = []
var_id_list = []
for val_dict,sample in zip(val_list,val_sample_list): 
    DataIn  = sample.Data[:,0:lastInputIndex+1]
    DataOut = sample.Data[:,lastInputIndex+1:]
    expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)
    sample_id = val_dict["id"].split(".")
    var_id_list.append(sample_id[0])

    time_list.append(val_dict["time"])
    recon_list.append(expected_data)
    gt_list.append(DataOut)
    id_list.append(val_dict["id"])

# save the results
create_dir(f"{subject_path}/repro")
save_results(session_data["subject_id"],time_list,gt_list,recon_list,train_mean_per_var,id_list,f"{subject_path}/repro/val")

# plot validation recon & gt with the training data's mean and CI
stats_fig,stats_ax = compare_multi_dim_data(
    x_list=time_list+time_list,
    data_list=gt_list+recon_list,
    dim=10,
    labels=[f"{i}.GT" for i in id_list]+[f"{i}.Recon" for i in id_list],
    xtype="time",
    datatype="tau",
    fig_label="Val Compare",
    split=1,
    legend=False
)
plot_mean_ci(
    time_list[0],
    gt_list_per_var,
    fig=stats_fig,
    axs=stats_ax,
    labels=[f"{x}.Train GT" for x in unique_var_id_list],
    relimit=False,
    split=1
)
interactive_plot(stats_fig,stats_ax)

"""
perform reconstruction for test task parameters
"""
recon_list = []
gt_list = []
id_list = []
time_list = []
var_id_list = []
for test_dict,sample in zip(test_list,test_sample_list): 
    DataIn  = sample.Data[:,0:lastInputIndex+1]
    DataOut = sample.Data[:,lastInputIndex+1:]
    expected_data,exp_sigma,_,_ = tpgmm.repro_condition_gmr(DataIn,sample,lastInputIndex,lastOutputIndex,DS=False,new_dt=0)
    sample_id = test_dict["id"].split(".")
    var_id_list.append(sample_id[0])

    time_list.append(test_dict["time"])
    recon_list.append(expected_data)
    gt_list.append(DataOut)
    id_list.append(test_dict["id"])
gt_list_per_var,unique_var_id_list = split_plot_all(var_id_list,time_list,gt_list,id_list,rep_split=4,fig_label="Test GT")
recon_list_per_var,_ = split_plot_all(var_id_list,time_list,recon_list,id_list,rep_split=4,fig_label="Test Recon")

# get the mean of each test validation, and repeat for 4 repetitions
test_mean_per_var = []
for samples_per_var in gt_list_per_var:
    mean = np.mean(np.array(samples_per_var),axis=0)
    for i in range(session_data["num_rep"]):
        test_mean_per_var.append(mean)
# save the results
save_results(session_data["subject_id"],time_list,gt_list,recon_list,test_mean_per_var,id_list,f"{subject_path}/repro/test")

# plot recon & gt's mean and ci for each var
stats_fig,stats_ax = compare_multi_dim_data(
    x_list=[],
    data_list=[],
    dim=10,
    labels=[],
    xtype="time",
    datatype="tau",
    fig_label="Test Compare",
    split=4,
    legend=False
)
plot_mean_ci(
    time_list[0],
    gt_list_per_var+recon_list_per_var,
    fig=stats_fig,
    axs=stats_ax,
    labels=[f"{x}.Test GT" for x in unique_var_id_list]+[f"{x}.Test Recon" for x in unique_var_id_list],
    relimit=True,
    split=4
)
interactive_plot(stats_fig,stats_ax)

""" END OF PIPELINE"""
end = times.perf_counter()
print(f"Fast Deploy:Elapsed \t= {(end - start):.4f}s")

