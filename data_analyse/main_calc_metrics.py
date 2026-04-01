import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from metrics_pkg import compute_norm_error,compute_norm_tau_peak_diff,q,compute_coverage
from data_process.file_util_pkg import load_npy,save_npy

def print_string(str_list):
    string = ""
    for i in str_list:
        string+=i
        string+=" "
    print(string)
def sanity_check(sample):
    # sanity check purposes   
    print_string(["-----",sample["subject_id"],sample["var-id-case"],"-----"])
    # print()
    print_string([f"{'metric':10}",
                   f"{'joint':15}",
                   f"{'diff':10}"])
    for metric in ["avg_norm_error","norm_diff_tau","coverage"]:
        for i,joint in enumerate(q):
            print_string([f"{metric:10}",
                           f"{joint:15}",
                           f"{sample[metric][i]:10.5f}"])
        print()
    print()
    
session_data = {
    "exp_id":"exp1",
    "patient_id":"p1",
    "subject_id":"ying2",
    "sbmvmt_num":2,
    "num_rep":4,
    "variants":["var_1","var_2","var_3","var_4","var_5","var_6"] #
}
subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')

val_samples_compile = []
test_samples_compile = []
for combi_num in range(6):
    for sample_num in range(4):
        # calculate metrics for validation set
        val_samples = load_npy(f"{subject_path}/repro/val_{combi_num}_{sample_num}.npy")
        for sample in val_samples:
            sample["avg_norm_error"] = compute_norm_error(sample["recon"],sample["compare"])
            sample["norm_diff_tau"] = compute_norm_tau_peak_diff(sample["recon"],sample["compare"])
            sample["coverage"] = compute_coverage(sample["recon"],sample["compare"])
            sanity_check(sample)
        val_samples_compile += val_samples
        
        # calculate metrics for test set
        test_samples = load_npy(f"{subject_path}/repro/test_{combi_num}_{sample_num}.npy")
        for sample in test_samples:
            sample["avg_norm_error"] = compute_norm_error(sample["recon"],sample["compare"])
            sample["norm_diff_tau"] = compute_norm_tau_peak_diff(sample["recon"],sample["compare"])
            sample["coverage"] = compute_coverage(sample["recon"],sample["compare"])
            sanity_check(sample)
        test_samples_compile += test_samples

save_npy(f"{subject_path}/repro/val_processed",val_samples_compile)
save_npy(f"{subject_path}/repro/test_processed",test_samples_compile)