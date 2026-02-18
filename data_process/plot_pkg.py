import matplotlib.pyplot as plt
import numpy as np

import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import data_analyse.metrics_pkg

""" split all variations / repetitions into a list of variations of a list of repetitions"""
def split_reps(data_list:list=[],n=4):
	result = [data_list[i:i+n] for i in range(0, len(data_list), n)]
	return result

""" get muliti color """
def get_n_colors(n_colors:int,split=4):
    cmap = plt.colormaps['Set1']
    # If n_colors is a multiple of split, get n/split colors and repeat each split times
    if n_colors > 0 and n_colors % split == 0 and n_colors/4 > 1:
        base_n = n_colors // split
        base_colors = [cmap(i / base_n) for i in range(base_n)]
        colors = [color for color in base_colors for _ in range(split)]
    else:
        colors = [cmap(i / (n_colors)) for i in range(n_colors)]
    return colors

""" hide / show plot interactively (click to hide/show) """
def interactive_plot(fig, axs):
    handles, labels = axs[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower right', ncol=4, draggable=True)
    
    # Map legend handles to original artists across all axes
    lined = {}
    for leghandle, label in zip(leg.legend_handles, labels):
        leghandle.set_picker(5)  # Works for both Line2D and Patch objects
        lined[leghandle] = []
        for ax in axs:
            # Check lines
            for line in ax.get_lines():
                if line.get_label() == label:
                    lined[leghandle].append(line)
            # Check patches (e.g., bar plots)
            for patch in ax.patches:
                if patch.get_label() == label:
                    lined[leghandle].append(patch)
            # Check collections (e.g., fill_between creates PolyCollection)
            for coll in ax.collections:
                if coll.get_label() == label:
                    lined[leghandle].append(coll)
    
    def on_pick(event):
        legline = event.artist
        if legline in lined:
            origlines = lined[legline]
            vis = not origlines[0].get_visible() if origlines else True
            for origline in origlines:
                origline.set_visible(vis)
            legline.set_alpha(1.0 if vis else 0.2)
            fig.canvas.draw()
    
    def on_click(event):
        # Right-click (button 3) to hide all
        if event.button == 3:
            for leghandle, origlines in lined.items():
                for origline in origlines:
                    origline.set_visible(False)
                leghandle.set_alpha(0.2)
            fig.canvas.draw()
    
    def on_resize(event):
        fig.tight_layout()
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('resize_event', on_resize)
    return leg

""" plot confidence interval with mean and std for n-lists list (each element in the list is a list of  2d arrays)"""
def plot_mean_ci(x:np.ndarray, data_list:list, fig=None,axs=None,labels:list=[],legend=True,relimit=False,dist="gaussian",split=4):

    colors = get_n_colors(len(data_list),split)
    for data,color,label in zip(data_list,colors,labels):
        if dist == "gaussian":
            mean = np.mean(np.array(data), axis=0)
            std = np.std(np.array(data), axis=0, ddof=0)  # ddof=1 for sample std
            n = np.array(data).shape[0]  # number of repetitions

            # 95% CI using t-distribution
            sem = std / np.sqrt(n)  # standard error of mean
            from scipy import stats
            t_crit = stats.t.ppf(0.975, df=n-1) 
            ci_95 = 1.96 * sem

            lower = mean - ci_95
            upper = mean + ci_95
            for i, ax in enumerate(axs):
                ax.plot(x, mean[:, i], ls='-',color=color, label=f"mean {label}", alpha=0.7, linewidth=3)
                # Plot 95% CI band
                ax.fill_between(x, lower[:, i], upper[:, i], 
                                alpha=0.3, color=color, label=f'95% CI {label}')
        else:
            # print("iqr")
            Q1 = np.percentile(np.array(data), 25,axis=0)  # 25th percentile
            Q2 = np.percentile(np.array(data), 50,axis=0)  # Median (50th percentile)
            Q3 = np.percentile(np.array(data), 75,axis=0)  # 75th percentile

            for i, ax in enumerate(axs):
                ax.plot(x, Q2[:, i], ls='-',color=color, label=f"mean {label}", alpha=0.7, linewidth=3)
                # Plot 95% CI band
                ax.fill_between(x, Q1[:, i], Q3[:, i], 
                                alpha=0.3, color=color, label=f'95% CI {label}')
    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels,loc='lower right', ncol=4, draggable=True)

"""compare multi dim data function individually from multiple sources"""
def compare_multi_dim_data(x_list:list,data_list:list,
                           dim:int,labels:list,xtype:str,datatype:str,
                           split=4,
                           sharex:bool=True,semilogx:bool=False,legend:bool=True,
                           fig_label:str="Take1",
                           show_stats:bool=False,show_zero_cross:bool=False):
    # init fig, axes
    fig = plt.figure(figsize=(15, 5),num=fig_label,tight_layout=True)
    axs = []
    colors = get_n_colors(len(x_list),split)

    # adjust the dimension labels for specific dimensions
    if dim == 1:
        dim_labels = [f"{datatype}"]
    elif dim == 3:
        dim_labels = [f"{datatype}_{i}" for i in ["X","Y","Z"]]
    elif dim == 6:
        dim_labels = [f"{datatype}_{i}" for i in ["FX","FY","FZ","MX","MY","MZ"]]
    elif dim == 18:
        dim_labels = [f"rft{j}_{datatype}_{i}" for j in range(3) for i in ["FX","FY","FZ","MX","MY","MZ"]]
    elif dim == 12:
        dim_labels = [f"{datatype}_{i}" for i in ['trunk_ie','trunk_aa','trunk_fe',
                        'clav_dep_ev','clav_prot_ret',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps','wrist_fe','wrist_dev']]
    else:
        dim_labels = [f"{datatype}_{i}" for i in ['trunk_ie','trunk_aa','trunk_fe',
                        'clav_dep_ev','clav_prot_ret',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps']]
        
    # initialize the col, rows, limits, titles, labels etc
    if datatype == 'q' or datatype == 'q_rad':
        limits =[
            (-50, 50), # trunk ie
            (-40, 40), # trunk aa
            (-30, 80), # trunk fe
            (-30, 30), # clav dep-elev
            (-15, 15), # clav pro retraction
            (-90, 180), # shoulder fe
            (-40, 180), # shoulder aa
            (-45, 90), # shoulder ie
            (-10, 170), # elbow fe
            (-30, 190) # elbow ps
        ]
    elif len(data_list) == 1:
        temp_arr = np.array(data_list)
        limits = [np.min(temp_arr),np.max(temp_arr)]
    elif len(data_list) > 1:
        temp_arr = np.array(data_list)
        limits = [np.min(temp_arr),np.max(temp_arr)]


    for i in range(dim):
        if dim >= 3:
            axs.append(fig.add_subplot(int(np.ceil(dim/3)),3,i+1))
        else:
            axs.append(fig.add_subplot(1,dim,i+1))

        if len(data_list) > 0:
            if datatype == "q":
                axs[i].set_ylim(limits[i][0],limits[i][1])
            elif datatype == "q_rad":
                axs[i].set_ylim(np.deg2rad(limits[i][0]),np.deg2rad(limits[i][1]))
            else:
                axs[i].set_ylim(limits[0]-0.1*abs(limits[0]),limits[1]+0.1*abs(limits[1]))

        axs[i].set_xlabel(f'{xtype}')
        axs[i].set_ylabel(f'{dim_labels[i]}')
        axs[i].set_title(f'{dim_labels[i]} vs {xtype}')
        axs[i].grid(True)
            
    # plot actual data
    for i,(x_array_i,data_array_i,label,color) in enumerate(zip(x_list, data_list, labels, colors)):
        if sharex:
            x_array_i = np.column_stack([x_array_i for i in range(dim)])

        if data_array_i.shape[1] != dim:
            data_array_i = np.column_stack([data_array_i for i in range(dim)])

        linewidth = 1
        alpha = 0.7
        for j,ax in enumerate(axs):
            if semilogx:
                ax.semilogx(x_array_i[:, j], data_array_i[:, j], ls='-',color=color, label=label, alpha=alpha, linewidth=linewidth)
            else:
                ax.plot(x_array_i[:, j], data_array_i[:, j], ls='-',color=color, label=label, alpha=alpha, linewidth=linewidth)
    
    # plot stats
    if show_stats:
        plot_mean_ci(x_list[0],[data_list], fig=fig,axs=axs,labels=["data"],legend=False,dist="iqr")

    # sanity check for zero crossings cuz why not
    for j,ax in enumerate(axs):
        if show_zero_cross:
            plot_velocity_zero_crossings(x_array_i[:, j], data_array_i[:, j], ax)
    
    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels,loc='lower right', ncol=4, draggable=True)

    plt.tight_layout()
    return fig,axs

"""plot 3d traj function"""
def plot_3d_trajectory(traj_list:list,label_list:list,fig=None,ax=None):
    n_colors = len(traj_list)
    colors = get_n_colors(n_colors)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(15, 5), subplot_kw={'projection': '3d'},num="3D Trajectory",tight_layout=True)

        ax.set_xlim([-0.9, 0.9])
        ax.set_ylim([-0.9, 0.9])
        ax.set_zlim([-0.2, 1.6])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')

    for traj,label,color in zip(traj_list,label_list,colors):
        ax.plot(traj[:, 0], 
            traj[:, 1], 
            traj[:, 2], 
            label=label, alpha=0.6, linewidth=1, color=color)
    ax.legend()
    plt.tight_layout()
    if fig is None or ax is None:
        return fig,ax
    else:
        return fig,ax
    
"""plot 3d points function"""
def plot_3d_points(traj_list:list,label_list:list,fig=None,ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(15, 5), subplot_kw={'projection': '3d'},tight_layout=True)

        ax.set_xlim([-0.9, 0.9])
        ax.set_ylim([-0.9, 0.9])
        ax.set_zlim([-0.2, 1.6])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')

    for traj,label in zip(traj_list,label_list):
        ax.scatter(traj[:, 0], 
            traj[:, 1], 
            traj[:, 2], 
            label=label, alpha=0.6, linewidth=1)
    ax.legend()
    plt.tight_layout()
    if fig is None or ax is None:
        return fig,ax
    else:
        return fig,ax

""" plot submovements for a single 3d trajectory"""
def plot_3d_submovements(traj,sbmvmt_indices):
    fig, ax = plt.subplots(figsize=(15, 5), subplot_kw={'projection': '3d'},tight_layout=True)
    ax.set_xlim([-0.9, 0.9])
    ax.set_ylim([-0.9, 0.9])
    ax.set_zlim([-0.2, 1.6])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')

    x = traj[:,0]
    y = traj[:,1]
    z = traj[:,2]

    n_colors = len(sbmvmt_indices)
    colors = get_n_colors(n_colors)

    for i,sbmvmt_index in enumerate(sbmvmt_indices):
        ax.plot(x[sbmvmt_index],y[sbmvmt_index],z[sbmvmt_index],marker='o', linestyle='None', color=colors[i],label=f'point {i+1}')
        start = sbmvmt_index
        if i < len(sbmvmt_indices)-1:
            end = sbmvmt_indices[i+1]
        else:
            end = -1
        ax.plot(x[start:end],y[start:end],z[start:end],linestyle='-', color=colors[i])
    ax.legend()
    plt.tight_layout()

""" plot zero velocity crossings for 1D velocity data"""
def plot_velocity_zero_crossings(time, velocity, ax):
    """ find zero crossing points in 1D velocity data and return their indices and times"""
    def find_zero_velocity_crossings(velocity: np.ndarray, time: np.ndarray = None):
        """
        Find indices where velocity crosses zero (sign change).
        
        Args:
            velocity: 1D array of velocity values
            time: Optional time array for interpolating exact crossing times
            
        Returns:
            crossing_indices: Indices where zero crossings occur
            crossing_times: Interpolated times (if time provided), else same as indices
        """
        # Find sign changes
        sign_changes = np.where(np.diff(np.sign(velocity)))[0]
        
        crossing_indices = []
        crossing_times = []
        
        for idx in sign_changes:
            crossing_indices.append(idx)
            
            # Interpolate exact crossing time
            if time is not None:
                t_cross = time[idx] + (time[idx + 1] - time[idx]) * (-velocity[idx]) / (velocity[idx + 1] - velocity[idx])
                crossing_times.append(t_cross)
            else:
                crossing_times.append(idx)
        
        return crossing_indices, crossing_times

    """Plot velocity and mark zero crossings."""
    # Find and mark crossings
    indices, times = find_zero_velocity_crossings(velocity, time)
    
    for t_cross in times:
        ax.axvline(t_cross, color='red', linestyle='-', alpha=0.7)

def split_plot_all(var_id_list,time_list,data_list,label_list,rep_split=4,data_type="tau",fig_label="Train"):
    # isolate each variation
    unique_var_id = []
    for x in var_id_list:
        if x in unique_var_id:
            continue
        else:
            unique_var_id.append(f"{x}")
    data_list_sep_var = split_reps(data_list,rep_split)
    # plot gt's mean and ci for each var
    fig,ax = compare_multi_dim_data(
        x_list=time_list,
        data_list=data_list,
        dim=data_list[0].shape[1],
        labels=label_list,
        xtype="time",
        datatype=data_type,
        fig_label=f"{fig_label}",
        legend=False,
        split=rep_split
    )
    plot_mean_ci(
        time_list[0],
        data_list_sep_var,
        fig=fig,
        axs=ax,
        labels=[f"{fig_label} {x}" for x in unique_var_id],
        relimit=True,
        split=rep_split
    )
    interactive_plot(fig,ax)
    return data_list_sep_var,unique_var_id