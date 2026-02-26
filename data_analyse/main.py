import sys,os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from metrics_pkg import compute_tau_peak,compute_impulse,compute_dtw,print_results
from data_visual.plot_pkg import compare_multi_dim_data
from data_process.file_util_pkg import load_npy,save_npy

def sanity_check(sample,item):
    # sanity check purposes
    from metrics_pkg import q
    # compare_multi_dim_data(
    #     [sample["time"],sample["time"]],
    #     [sample["compare"],sample[item]],
    #     dim = 10,
    #     labels=["mean",item],xtype="time",datatype="tau",
    #     split=1,
    #     fig_label=f"sanity_check_{item}_{sample['id']}"
    #     )
    
    print("-----",item,"-----")
    print_results([f"{'metric':10}",
                   f"{'joint':15}",
                   f"{'mean value':15}",
                   f"{item+' value':15}",
                   f"{'diff':10}"])
    for metric in ["tau_peak","impulse"]:
        for i,joint in enumerate(q):
            print_results([f"{metric:10}",
                           f"{joint:15}",
                           f"{sample['metric']['compare'][metric][i]:15.4f}",
                           f"{sample['metric'][item][metric][i]:15.4f}",
                           f"{sample['diff'][item][metric][i]:10.4f}"])
            
            # print("tau peak",met,sample["diff"][item]["tau_peak"][i])
        print()
    print_results([f"{'DTW':10}",
                   f"{'-':15}",
                   f"{'-':15}",
                   f"{'-':15}",
                   f"{sample['diff'][item]['DTW']:10.4f}"])
    print()
    
session_data = {
    "subject_id":"exp1/p1/ying",
    "task_id":"task_1",
    "sbmvmt_num":2,
    "num_rep":4,
    "variants":["var_1","var_2","var_3","var_4","var_5","var_6"]
}
subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["subject_id"]}/{session_data["task_id"]}')

#%%
# calculate metrics and diff for validation set
val_samples = load_npy(f"{subject_path}/repro/val.npy")
for sample in val_samples:
    # calculate metric
    sample["metric"] = {}
    for item in ["compare","thera","recon"]:
        # calculate tau_peak
        tau_peak = compute_tau_peak(sample[item])
        # calculate impulse
        impulse = compute_impulse(sample["time"],sample[item])
        sample["metric"][item] = {
            "tau_peak":tau_peak,
            "impulse":impulse
            }
        
    # calculate deviation from comparator's metric
    sample["diff"] = {}
    print("=====",sample['subject_id'],"=====",f"{sample['var']}.{sample['id']}.{sample['case']}","=====")
    for item in ["thera","recon"]:
        sample["diff"][item] = {
            "tau_peak":np.abs((sample["metric"][item]["tau_peak"]-sample["metric"]["compare"]["tau_peak"])/sample["metric"]["compare"]["tau_peak"])*100,
            "impulse":np.abs((sample["metric"][item]["impulse"]-sample["metric"]["compare"]["impulse"])/sample["metric"]["compare"]["impulse"])*100,
            "DTW": compute_dtw(sample["compare"],sample[item])[0]
            }
        sanity_check(sample,item)
        
# calculate metrics and diff for test set
test_samples = load_npy(f"{subject_path}/repro/test.npy")
for sample in test_samples:
    # calculate metric
    sample["metric"] = {}
    for item in ["compare","thera","recon"]:
        # calculate tau_peak
        tau_peak = compute_tau_peak(sample[item])
        # calculate impulse
        impulse = compute_impulse(sample["time"],sample[item])
        sample["metric"][item] = {
            "tau_peak":tau_peak,
            "impulse":impulse
            }
        
    # calculate deviation from comparator's metric
    sample["diff"] = {}
    print("=====",sample['subject_id'],"=====",f"{sample['var']}.{sample['id']}.{sample['case']}","=====")
    for item in ["thera","recon"]:
        sample["diff"][item] = {
            "tau_peak":np.abs((sample["metric"][item]["tau_peak"]-sample["metric"]["compare"]["tau_peak"])/sample["metric"]["compare"]["tau_peak"])*100,
            "impulse":np.abs((sample["metric"][item]["impulse"]-sample["metric"]["compare"]["impulse"])/sample["metric"]["compare"]["impulse"])*100,
            "DTW": compute_dtw(sample["compare"],sample[item])[0]
            }
        sanity_check(sample,item)
     
# resave results
all_samples = val_samples+test_samples
save_npy(f"{subject_path}/repro/all_processed",all_samples)

