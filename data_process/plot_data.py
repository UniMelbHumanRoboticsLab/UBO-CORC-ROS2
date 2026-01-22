import matplotlib.pyplot as plt
import numpy as np

""" get muliti color """
def get_n_colors(n_colors:int):
    cmap = plt.colormaps['tab10']
    colors = [cmap(i / (n_colors)) for i in range(n_colors)]
    return colors

"""compare multi dim data function individually from multiple sources"""
def compare_multi_dim_data(x_list:list,data_list:list,dim:int,labels:list,xtype:str,datatype:str,sharex:bool=True,semilogx:bool=False,fig_label:str="Take1"):
    fig = plt.figure(figsize=(15, 5),num=fig_label)

    axs = []
    dim_labels = []
    if dim >= 3:
        for i in range(dim):
            axs.append(fig.add_subplot(int(np.ceil(dim/3)),3,i+1))
            dim_labels.append(f"{datatype}_{i}")
    else:
        for i in range(dim):
            axs.append(fig.add_subplot(1,dim,i+1))
            dim_labels.append(f"{datatype}_{i}")

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
                        'scapula_de','scapula_pr',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps']]

    n_colors = len(x_list)
    colors = get_n_colors(n_colors)

    for x_array_i,data_array_i,label,color in zip(x_list, data_list, labels, colors):
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
            ax.legend()
            ax.grid(True)
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
