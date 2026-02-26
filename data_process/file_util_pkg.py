import pandas as pd
import numpy as np
import os 

""" compile raw data"""
def get_raw_data(data_path):
	data = pd.read_csv(data_path)
	time_data = data["elapsed_time"].values
	corc_data = data[["F1x", "F1y", "F1z", "T1x", "T1y", "T1z",
			"F2x", "F2y", "F2z", "T2x", "T2y", "T2z",
			"F3x", "F3y", "F3z", "T3x", "T3y", "T3z"]].values
	joints = ['trunk_ie','trunk_aa','trunk_fe',
				'clav_dep_ev','clav_prot_ret',
				'shoulder_fe','shoulder_aa','shoulder_ie',
				'elbow_fe','elbow_ps',
				'wrist_fe','wrist_dev']
	right_xsens_data = data[[f"{joint}_right" for joint in joints]].values

	return data,time_data,corc_data,right_xsens_data

""" compile processed data into train test val"""
def get_processed_data(data_path,degree_flag=False):
    q = ['trunk_ie','trunk_aa','trunk_fe','clav_dep_ev','clav_prot_ret','shoulder_fe','shoulder_aa','shoulder_ie','elbow_fe','elbow_ps']
    qdot = [f"{joint}_dot" for joint in q]
    tau = [f"tau_{joint}" for joint in q]
    
    data = pd.read_csv(data_path)
    time_data = data["norm_time"].values
    indices = data["index"].values
    if degree_flag:
        joint_kinematics = np.rad2deg(data[q+qdot].values) # change back to degrees to check
    else:
        joint_kinematics = data[q+qdot].values # change back to degrees to check
    joint_torques = data[tau].values

    return time_data,indices,joint_kinematics,joint_torques
def compile_train_val_test_data(session_data,task_path,degree_flag=False):
	train_test_split = pd.read_csv(f'{task_path}/train_test_split.csv')["split"].values
	train_val_df = pd.read_csv(f'{task_path}/train_val_split.csv')
	train_val_split = dict(zip(train_val_df["repetition"], train_val_df["split"]))

	train_list = []
	val_list = []
	test_list = []

	for case,var in zip(train_test_split,session_data["variants"]):
		reps = range(1,session_data["num_rep"]+1)
		for rep in reps:
			time,_,q_qdot,tau	= get_processed_data(f'{task_path}/{var}/processed/UBORecord{rep}Log.csv',degree_flag)
			tp 				= pd.read_csv(f'{task_path}/{var}/processed/tp/UBOTP{rep}Log.csv').values
			if degree_flag:
				tp[:,900:920] = np.rad2deg(tp[:,900:920])  # change back to degrees to check
			var_rep = {
			"time":time,
			"data":np.hstack((q_qdot,tau)),
			"tp":tp
			}

			if case == "train":
				# check if var_rep is train or val
				split = train_val_split[f"{var}/processed/UBORecord{rep}Log.csv"]
				var_rep["id"] = f"{var}.{rep}.{split}"
				if split == "train":
						train_list.append(var_rep)
				elif split == "val":
						val_list.append(var_rep)
			elif case == "test":
				# test list
				var_rep["id"] = f"{var}.{rep}.{case}"
				test_list.append(var_rep)
	return train_list,val_list,test_list

""" select which variation for train and for test """
def separate_train_test(variant_list,path):
    # Given numbers
    numbers = np.array([i for i in range(len(variant_list))])

    # Randomly select 2 numbers for test
    diff = 0
    while diff < 3 or diff == len(numbers)-1:
        test = np.random.choice(numbers, size=2, replace=False)+1
        diff = test[1]-test[0] # make sure that selected points are always 2 points away and not simultaneously the 2 extremes

    case_id = []
    train_var = []
    test_var = []
    # Iterate through the files in the source folder
    for var in variant_list:
        # Use regex to extract the number between 'm' and 'd'
        var_id = int(var[-1])
        # Check if m_value is even or odd
        if var_id in test:
            case_id.append("test")
            test_var.append(var)
        else:
            case_id.append("train")
            train_var.append(var)

    # save train/test split to csv
    train_test_df = pd.DataFrame({
        "variant": variant_list,
        "split": case_id
    })
    train_test_df.to_csv(f'{path}',index=True)
    return case_id,train_var,test_var

""" select which repetition for train and for validation """
def separate_train_val(train_list, repetition_num, path):
    train_reps = []
    val_reps = []
    case_id = []
    repetitions = []
    # randomly select one rep for validation, rest for training
    for var in train_list:
        val_rep_num = np.random.randint(1, repetition_num + 1)
        for rep in range(1, repetition_num + 1):
            rep_path = f"{var}/processed/UBORecord{rep}Log.csv"
            repetitions.append(rep_path)
            if rep == val_rep_num:
                val_reps.append(rep_path)
                case_id.append("val")
            else:
                train_reps.append(rep_path)
                case_id.append("train")
    
    # save train/val split to csv
    split_labels = ["train"] * len(train_reps) + ["val"] * len(val_reps)
    train_val_df = pd.DataFrame({
        "repetition": repetitions,
        "split": case_id
    })
    train_val_df.to_csv(f'{path}', index=False)
    
    return train_reps, val_reps

""" create directory if needed"""
def create_dir(data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

""" split all variations / repetitions into a list of variations of a list of repetitions"""
def split_reps(data_list:list=[],n=4):
	result = [data_list[i:i+n] for i in range(0, len(data_list), n)]
	return result

""" load / save npy files for each samples and convert them to list"""
def load_npy(data_path):
    return  np.load(data_path,allow_pickle=True).tolist()
def save_npy(data_path,compile_data):
    np.save(data_path, np.array(compile_data,dtype=dict))