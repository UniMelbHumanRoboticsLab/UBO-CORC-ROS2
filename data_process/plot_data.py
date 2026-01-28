import matplotlib.pyplot as plt
import numpy as np

""" get muliti color """
def get_n_colors(n_colors:int):
    cmap = plt.colormaps['tab10']
    colors = [cmap(i / (n_colors)) for i in range(n_colors)]
    return colors

"""compare multi dim data function individually from multiple sources"""
def compare_multi_dim_data(x_list:list,data_list:list,
                           dim:int,labels:list,xtype:str,datatype:str,
                           sharex:bool=True,semilogx:bool=False,
                           fig_label:str="Take1",
                           show_stats:bool=False,show_zero_cross:bool=False):
    # init fig, axes, and dims
    fig = plt.figure(figsize=(15, 5),num=fig_label)
    axs = []
    dim_labels = []
    n_colors = len(x_list)
    colors = get_n_colors(n_colors)

    # adjust the dimension labels for specific dimensions
    if dim == 1:
        dim_labels = [f"{datatype}"]
    elif dim == 3:
        dim_labels = [f"{datatype}_{i}" for i in ["X","Y","Z"]]
    elif dim == 6:
        dim_labels = [f"{datatype}_{i}" for i in ["FX","FY","FZ","MX","MY","MZ"]]
    elif dim == 18:
        dim_labels = [f"rft{j}_{datatype}_{i}" for j in range(3) for i in ["FX","FY","FZ","MX","MY","MZ"]]
    else:
        dim_labels = [f"{datatype}_{i}" for i in ['trunk_ie','trunk_aa','trunk_fe',
                        'clav_dep_ev','clav_prot_ret',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps']]
        
    # set the col , rows and the limits
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
            (-10, 190) # elbow ps
        ]
    else:
        temp_arr = np.array(data_list)
        limits = [np.min(temp_arr),np.max(temp_arr)]

    for i in range(dim):
        if dim >= 3:
            axs.append(fig.add_subplot(int(np.ceil(dim/3)),3,i+1))
        else:
            axs.append(fig.add_subplot(1,dim,i+1))
        if datatype == "q":
            axs[i].set_ylim(limits[i][0],limits[i][1])
        elif datatype == "q_rad":
            axs[i].set_ylim(np.deg2rad(limits[i][0]),np.deg2rad(limits[i][1]))
        else:
            axs[i].set_ylim(limits[0]-0.1*abs(limits[0]),limits[1]+0.1*abs(limits[1]))
        dim_labels.append(f"{datatype}_{i}")

    # plot stats
    if show_stats:
        mean = np.mean(np.array(data_list), axis=0)
        std = np.std(np.array(data_list), axis=0, ddof=1)  # ddof=1 for sample std
        n = np.array(data_list).shape[0]  # number of repetitions

        # 95% CI using t-distribution
        sem = std / np.sqrt(n)  # standard error of mean
        from scipy import stats
        t_crit = stats.t.ppf(0.975, df=n-1)  # ~4.303 for n=3
        ci_95 = t_crit * sem

        lower = mean - ci_95
        upper = mean + ci_95
        for i, ax in enumerate(axs):
            if semilogx:
                ax.semilogx(x_list[0], mean[:, i], ls='-',color="red", label="mean", alpha=0.7, linewidth=3)
            else:
                ax.plot(x_list[0], mean[:, i], ls='-',color="red", label="mean", alpha=0.7, linewidth=3)
            # Plot 95% CI band
            ax.fill_between(x_list[0], lower[:, i], upper[:, i], 
                            alpha=0.3, color='blue', label='95% CI')
            ax.legend()
            
    # plot actual data
    for i,(x_array_i,data_array_i,label,color) in enumerate(zip(x_list, data_list, labels, colors)):
        if sharex:
            x_array_i = np.column_stack([x_array_i for i in range(dim)])

        if data_array_i.shape[1] != dim:
            data_array_i = np.column_stack([data_array_i for i in range(dim)])

        linewidth = 1
        alpha = 0.7

        for j,(ax,dim_label) in enumerate(zip(axs,dim_labels)):
            if semilogx:
                ax.semilogx(x_array_i[:, j], data_array_i[:, j], ls='-',color=color, label=label, alpha=alpha, linewidth=linewidth)
            else:
                ax.plot(x_array_i[:, j], data_array_i[:, j], ls='-',color=color, label=label, alpha=alpha, linewidth=linewidth)

            ax.set_xlabel(f'{xtype}')
            ax.set_ylabel(f'{dim_label}')
            ax.set_title(f'{dim_label} vs {xtype}')
            ax.grid(True)

    # sanity check for zero crossings cuz why not
    for j,ax in enumerate(axs):
        if show_zero_cross:
            plot_velocity_zero_crossings(x_array_i[:, j], data_array_i[:, j], ax)
        else:
            ax.legend()
            # break

    
    plt.tight_layout()
    return fig,axs

"""plot 3d traj function"""
def plot_3d_trajectory(traj_list:list,label_list:list,fig=None,ax=None):
    n_colors = len(traj_list)
    colors = get_n_colors(n_colors)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(15, 5), subplot_kw={'projection': '3d'},num="3D Trajectory")

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
        fig, ax = plt.subplots(figsize=(15, 5), subplot_kw={'projection': '3d'})

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
    fig, ax = plt.subplots(figsize=(15, 5), subplot_kw={'projection': '3d'})
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