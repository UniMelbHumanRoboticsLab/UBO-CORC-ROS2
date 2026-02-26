import numpy as np
from scipy import integrate
from dtaidistance import dtw_ndim,dtw

q = ['trunk_ie','trunk_aa','trunk_fe',
	'clav_dep_ev','clav_prot_ret',
	'shoulder_fe','shoulder_aa','shoulder_ie',
	'elbow_fe','elbow_ps']

def print_results(item_list):
    string = ""
    for item in item_list:
        string+=f"{item}|"
    print(string)
        
def load_npy(data_path):
    np.load(data_path,allow_pickle=True).tolist()
    
def compute_impulse(time,torque):
    """ Compute impulse of each joint using trapezoidal integration """
    torque_abs = np.abs(torque)
    impulse = []
    for i in range(torque.shape[1]):
        impulse_q = integrate.simpson(torque_abs[:,i], x=time)
        # print("impulse",q[i],impulse_q)
        impulse.append(impulse_q)
    return np.array(impulse)

def compute_tau_peak(torque):
    """ Compute peak torque of each joint """
    peak_tau = np.max(np.abs(torque),axis=0)
    # for i in range(torque.shape[1]):
    #     print("tau peak",q[i],peak_tau[i])
    return peak_tau

# Evaluate Mean DTW Distance Cost for all demos and generated trajectory
def compute_dtw(ref_traj,target_traj):
    # convert to double data type
    ref_traj = ref_traj.astype(np.double)
    target_traj = target_traj.astype(np.double)
    
    if len(ref_traj.shape) == 1: # univariate
        distance, paths = dtw.warping_paths_fast(ref_traj, target_traj,keep_int_repr=True)
        best_path = dtw.best_path(paths)
        similarity_score = distance / len(best_path)
    elif len(ref_traj.shape) > 1: # multidim
        distance, paths = dtw_ndim.warping_paths_fast(ref_traj, target_traj,keep_int_repr=True)
        best_path = dtw.best_path(paths)
        similarity_score = distance / len(best_path)
        
    from dtaidistance import dtw_visualisation as dtwvis
    dtwvis.plot_warpingpaths(ref_traj, target_traj,paths,path=best_path,shownumbers=False,show_diagonal=True,showlegend=True)
    return similarity_score,distance

if __name__ == "__main__":
    # sanity test for all metrics
    time = 5*np.linspace(0,1,num=100)
    cs = np.array([np.cos(2*np.pi*1*time),np.sin(2*np.pi*1*time)]).T
    sc = np.array([np.sin(2*np.pi*1*time),np.cos(2*np.pi*1*time)]).T
    sc2 = 5*sc
    
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data_visual.plot_pkg import compare_multi_dim_data,interactive_plot
    
    fig,axis = compare_multi_dim_data(
        [time,time,time],
        [cs,sc,sc2],
        dim = 2,
        labels=["cs","sc","sc2"],xtype="time",datatype="sinu",
        split=1,
        fig_label=f"sanity_check_metric"
        )
    interactive_plot(fig,axis)
    
    print("COS-SIN set metrics")
    tau_peak = compute_tau_peak(cs) # expect 1
    for i,met in enumerate(["D1","D2"]):
        print("tau peak",met,tau_peak[i])
    print()
    impulse = compute_impulse(time,cs) # expected 3.18310506506369
    for i,met in enumerate(["D1","D2"]):
        print("tau peak",met,impulse[i])
    print()
    
    print("DTW check")
    sim1,dis1 = compute_dtw(cs,sc)
    print("CS-SC:",sim1,dis1 )
    sim2,dis2 = compute_dtw(cs,sc2)
    print("CS-SC*2:",sim2,dis2)
    sim3,dis3 = compute_dtw(sc,sc2)
    print("SC-SC*2:",sim3,dis3)