import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_pkg import compare_multi_dim_data,plot_3d_submovements
from scipy.interpolate import CubicSpline

""" separate variations into train test """
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

""" separate repetitions in training variations into train validation"""
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

"""low pass filter"""
def lpf(time_array,data,ts=0.01,fc=50,datatype="trajectory",plot_results=False):
    from scipy.fft import fft,fftfreq
    from scipy.signal.windows import blackman
    from scipy.signal import butter,sosfreqz,sosfiltfilt

    num_dtps = data.shape[0]
    dim = data.shape[1]
    filtered_data = []
    fft_arr = []
    fft_filt_arr = []
    db_arr = []

    # Define the sos filter
    
    fs = 1/ts  # Sampling frequency
    nyquist = fs / 2  # Nyquist frequency
    wc = fc / nyquist # Normalize cut-off frequency by Nyquist frequency (fs/2)

    xf = fftfreq(num_dtps, ts)[:num_dtps//2] # convert time to frequency domain
    window_fft = blackman(num_dtps) # define fft window
    sos = butter(4, wc, 'low', analog=False, output='sos')
    w, h = sosfreqz(sos, worN=8000, fs=fs)

    for i in range(dim):
        disturb = 0 * np.cos(2 * np.pi * 25 * time_array) # just to check the lpf
        curr_dim_data = data[:,i] + disturb

        # FFT resampled data
        rja_fft = fft(curr_dim_data*window_fft)
        fft_arr.append(np.abs(rja_fft[0:num_dtps//2]))
        # filter the data
        filtered_dim_data = sosfiltfilt(sos , curr_dim_data)
        filtered_data.append(filtered_dim_data)
        # FFT filtered data
        rja_fft_filt = fft(filtered_dim_data*window_fft)
        fft_filt_arr.append(np.abs(rja_fft_filt[0:num_dtps//2]))

        db_arr.append(np.abs(h))

    filtered_data = np.array(filtered_data).T
    fft_arr = np.array(fft_arr).T   
    fft_filt_arr = np.array(fft_filt_arr).T
    db_arr = np.array(db_arr).T

    if plot_results:
        compare_multi_dim_data(
            [xf, xf],
            [fft_arr, fft_filt_arr],
            dim,
            ['noisy', 'filtered'],
            'Frequency',
            f'FFT {datatype}',
            semilogx=True,
            fig_label=f'FFT_{datatype}'
        )
        compare_multi_dim_data(
            [w],
            [db_arr],
            dim,
            ['FreqResp'],
            'Frequency (Hz)',
            f'Frequency Response {datatype}',
            semilogx=True,
            fig_label=f'FreqResp_{datatype}'
        )
        compare_multi_dim_data(
            [time_array,time_array],
            [data, filtered_data],
            dim,
            ['noisy', 'filtered'],
            'Time',
            datatype,
            fig_label=f'LFP_{datatype}'    
        )
    return filtered_data

"""calculate fixed derivative using finite differences"""
def calc_fixed_diff(data, dt):
    derivative = np.diff(data, axis=0) / dt
    derivative = np.vstack([derivative, derivative[-1]])
    return derivative

"""calculate magnitude of vectors"""
def calc_mag(vectors):
    return np.linalg.norm(vectors, axis=1)[:,np.newaxis]

"""segment submovements for current repetition"""
def segment_sbmvmts(time_array,hand_traj,hand_speed,submovement_num,data_path,redo=True):

    if submovement_num == 1:
        segment_type = "StartEnd"
    else:
        segment_type = "Submovements"
        
    if os.path.isfile(data_path):
        print(f"{data_path[115:]} exists")
        # read sbmvmt_indices from txt file
        with open(data_path, 'r') as f:
            lines = f.readlines()
            sbmvmt_indices = np.array(lines[0].split(": ")[1].split(), dtype=int)
            print(f"{segment_type} Indices:", sbmvmt_indices)
        if redo:
            plot_3d_submovements(hand_traj,sbmvmt_indices=sbmvmt_indices)
            plt.show(block=False)
            if input("Good?:") == 'y':
                good = True
                plt.close("all")
            else:
                good = False  
                plt.close("all")
        else:
            good = True
    else:
        print(f"{data_path[115:]} DNE")
        good = False

    while not good:
        from mpl_point_clicker import clicker
        fig,axs = compare_multi_dim_data([time_array], [hand_speed], 1, ['speed'], 'Time', f'Hand Speed',fig_label= data_path)
        klicker = clicker(axs[0], ["event"], markers=["x"])
        plt.show(block=True)
        points = klicker.get_positions()['event']

        sbmvmt_indices = []
        for point in points:
            idx = (np.abs(time_array - point[0])).argmin()
            sbmvmt_indices.append(idx)
        if submovement_num != 1:
            sbmvmt_indices = [0] + sbmvmt_indices + [len(time_array)-1]

        if len(sbmvmt_indices) != submovement_num+1:
            print("Restart Selection:",len(sbmvmt_indices), "selected, need", submovement_num+1)
        else:
            print(f"Selected {segment_type} Indices:",sbmvmt_indices)

            plot_3d_submovements(hand_traj,sbmvmt_indices=sbmvmt_indices)
            plt.show(block=False)
            if input("Good?:") == 'y':
                good = True
                plt.close("all")
                sbmvmt_indices = np.array(sbmvmt_indices,dtype=int)
                indexed_time = np.array(time_array[sbmvmt_indices],dtype=float)
                
                # save to txt file
                with open(data_path, 'w') as f:
                    f.write("sbmvmt_indices: " + " ".join(map(str, sbmvmt_indices)) + "\n")
                    f.write("indexed_time: " + " ".join(map(str, indexed_time)) + "\n") 
            else:
                good = False  
                plt.close("all")
    
    if submovement_num != 1:
        indices_array = np.ones((time_array.shape[0],1))
        for i in range(len(sbmvmt_indices)-1):
            start_idx = sbmvmt_indices[i]
            end_idx = sbmvmt_indices[i+1]
            indices_array[start_idx:end_idx,0] = i+1
        indices_array[sbmvmt_indices[len(sbmvmt_indices)-1]] = i+1
    else:
        sbmvmt_indices[-1] = sbmvmt_indices[-1]+1
        indices_array = np.zeros((time_array.shape[0],1))
    print()
    return indices_array,sbmvmt_indices

""" up/down sample data to new time array"""
def rescale(t_old, data,t_new,datatype="trajectory",plot_results=False):
    interp = CubicSpline(t_old, data, axis=0,bc_type='natural')
    data_interp = interp(t_new)
    dim = data.shape[1]
    if plot_results:
        compare_multi_dim_data(
            [t_old,t_new],
            [data, data_interp],
            dim,
            ['old scale', 'new scale'],
            'Time',
            datatype,
            fig_label=f'rescale_{datatype}'    
        )
    return data_interp