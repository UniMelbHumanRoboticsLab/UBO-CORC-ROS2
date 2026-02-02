import pandas as pd
import numpy as np

q = ['trunk_ie','trunk_aa','trunk_fe',
	'clav_dep_ev','clav_prot_ret',
	'shoulder_fe','shoulder_aa','shoulder_ie',
	'elbow_fe','elbow_ps']


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

""" compile processed data and do data splitting"""
def get_processed_data(data_path,degree_flag=False):

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
				var_rep["id"] = f"{split}.{var}.{rep}"
				if split == "train":
						train_list.append(var_rep)
				elif split == "val":
						val_list.append(var_rep)
			elif case == "test":
				# test list
				var_rep["id"] = f"{case}.{var}.{rep}"
				test_list.append(var_rep)
	return train_list,val_list,test_list

""" compile results """
def get_results(data_path):
	tau_repro = [f"tau_{joint}_repro" for joint in q]
	tau = [f"tau_{joint}_gt" for joint in q]

	data = pd.read_csv(data_path)
	time_data = data["norm_time"].values
	joint_torques_gt = data[tau].values
	joint_torques_repro = data[tau_repro].values

	return time_data,joint_torques_gt,joint_torques_repro

def compile_val_test_repro(session_data,task_path):

	train_test_split = pd.read_csv(f'{task_path}/train_test_split.csv')["split"].values
	train_val_df = pd.read_csv(f'{task_path}/train_val_split.csv')
	train_val_split = dict(zip(train_val_df["repetition"], train_val_df["split"]))

	val_list = []
	test_list = []

	for case,var in zip(train_test_split,session_data["variants"]):
		reps = range(1,session_data["num_rep"]+1)
		for rep in reps:
			if case == "train":
				# check if var_rep is train or val
				split = train_val_split[f"{var}/processed/UBORecord{rep}Log.csv"]
				
				if split == "val":
					time,tau_gt,tau_repro	= get_results(f'{task_path}/{var}/processed/repro/UBORepro{rep}Log.csv')
					var_rep = {
					"time":time,
					"gt":tau_gt,
					"repro":tau_repro,
					"id" : f"{split}.{var}.{rep}"
					}
					val_list.append(var_rep)
			elif case == "test":
				# test list
				time,tau_gt,tau_repro	= get_results(f'{task_path}/{var}/processed/repro/UBORepro{rep}Log.csv')
				var_rep = {
				"time":time,
				"gt":tau_gt,
				"repro":tau_repro,
				"id":f"{case}.{var}.{rep}"
				}
				test_list.append(var_rep)
	return val_list,test_list