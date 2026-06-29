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
    print_string(["-----",sample["patient_id"],sample["subject_id"],sample["var-id-case"],"-----"])
    # print()
    print_string([f"{'metric':10}",
                   f"{'joint':15}",
                   f"{'diff':10}"])
    for cur_case in ["gt","recon","recon_lut"]:
        for metric in ["avg_norm_error","norm_diff_tau","coverage"]:
            for i,joint in enumerate(q):
                case_metric = f"{cur_case}_{metric}"
                print_string([f"{case_metric:10}",
                               f"{joint:15}",
                               f"{sample[case_metric][i]:10.5f}"])
        print()
    print()

for p in range(1,4):
    if p == 1:
        sm_num = 2
    else:
        sm_num=4
        
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
        
        val_samples_compile = []
        test_samples_compile = []
        for combi_num in range(6):
            for sample_num in range(4):
                # calculate metrics for validation set
                val_samples = load_npy(f"{subject_path}/repro/val_{combi_num}_{sample_num}.npy")
                for sample in val_samples:
                    print(f"{sample['patient_id']}_{sample['subject_id']}_{combi_num}_{sample_num}_{sample['var-id-case']}")
                    for i in ["gt","recon","recon_lut"]:
                        sample[i+"_avg_norm_error"] = compute_norm_error(sample[i],sample["compare"])
                        sample[i+"_norm_diff_tau"] = compute_norm_tau_peak_diff(sample[i],sample["compare"])
                        sample[i+"_coverage"] = compute_coverage(sample[i],sample["compare"])
                    sanity_check(sample)
                val_samples_compile += val_samples
                
                # calculate metrics for test set
                test_samples = load_npy(f"{subject_path}/repro/test_{combi_num}_{sample_num}.npy")
                for sample in test_samples:
                    print(f"{sample['patient_id']}_{sample['subject_id']}_{combi_num}_{sample_num}_{sample['var-id-case']}")
                    for i in ["gt","recon","recon_lut"]:
                        sample[i+"_avg_norm_error"] = compute_norm_error(sample[i],sample["compare"])
                        sample[i+"_norm_diff_tau"] = compute_norm_tau_peak_diff(sample[i],sample["compare"])
                        sample[i+"_coverage"] = compute_coverage(sample[i],sample["compare"])
                    sanity_check(sample)
                test_samples_compile += test_samples
        
        save_npy(f"{subject_path}/repro/val_processed",val_samples_compile)
        save_npy(f"{subject_path}/repro/test_processed",test_samples_compile)
