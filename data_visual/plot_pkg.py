import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 22})
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats
import numpy as np

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_process.file_util_pkg import split_reps
from data_analyse.stats_pkg import compute_central_tendency


def split_plot_all(var_id_list,time_list,data_list,label_list,rep_split=4,data_type="tau",fig_label="Train",plot=False):
    # isolate each variation
    unique_var_id = []
    for x in var_id_list:
        if x in unique_var_id:
            continue
        else:
            unique_var_id.append(f"{x}")
            
    data_list_per_var = split_reps(data_list,rep_split)
    # plot gt's mean and ci for each var
    if plot:
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
        plot_stats(
            time_list[0],
            data_list_per_var,
            fig=fig,
            axs=ax,
            labels=[f"{x}.{fig_label}" for x in unique_var_id],
            relimit=True,
            split=rep_split
        )
        interactive_plot(fig,ax)
    return data_list_per_var,unique_var_id

""" get muliti color """
def get_n_colors(n_colors:int,split=4,shuffle=False):
    if shuffle:
        cmap = plt.colormaps['Dark2']
    else:
        cmap = plt.colormaps['Set1']  
    # If n_colors is a multiple of split, get n/split colors and repeat each split times
    if n_colors > 0 and n_colors % split == 0 and n_colors/split > 1:
        base_n = n_colors // split
        base_colors = [cmap(i) for i in range(base_n)]
        colors = [color for color in base_colors for _ in range(split)]
    else:
        colors = [cmap(i / (n_colors)) for i in range(n_colors)]
    return colors

""" plot confidence interval with mean and std for n-lists list (each element in the list is a list of  2d arrays)"""
def plot_stats(x:np.ndarray, data_list:list, fig=None,axs=None,labels:list=[],legend=True,relimit=False,dist="gaussian",split=4):
    if relimit:
        temp_arr = np.array(data_list)
        limits = [np.min(temp_arr),np.max(temp_arr)]
        for i, ax in enumerate(axs):
            axs[i].set_ylim(limits[0]-0.1*abs(limits[0]),limits[1]+0.1*abs(limits[1]))

    colors = get_n_colors(len(data_list),split)
    for data,color,label in zip(data_list,colors,labels):
        if dist == "gaussian":
            
            mean,median,max,min = compute_central_tendency(data)
            
            # mean = np.mean(np.array(data), axis=0)
            # std = np.std(np.array(data), axis=0, ddof=0)  # ddof=1 for sample std
            # n = np.array(data).shape[0]  # number of repetitions
            # # 95% CI using t-distribution
            # sem = std / np.sqrt(n)  # standard error of mean
            # from scipy import stats
            # t_crit = stats.t.ppf(0.975, df=n-1) 
            # ci_95 = t_crit * sem  # ✅ Margin of error           
            # u = mean + ci_95
            # l = mean - ci_95
            
            for i, ax in enumerate(axs):
                # plot median
                ax.plot(x, median[:, i], ls=':',color=color, label=f"{label}.median", alpha=1, linewidth=1)
                # Plot min max bound
                ax.fill_between(x, min[:, i], max[:, i], 
                                alpha=0.6, color=color, label=f'{label}.Bound')
        else:
            print("iqr")
            Q1 = np.percentile(np.array(data), 25,axis=0)  # 25th percentile
            Q2 = np.percentile(np.array(data), 50,axis=0)  # Median (50th percentile)
            Q3 = np.percentile(np.array(data), 75,axis=0)  # 75th percentile

            for i, ax in enumerate(axs):
                ax.plot(x, Q2[:, i], ls='-',color=color, label=f"{label}.mean", alpha=0.7, linewidth=3)
                # Plot 95% CI band
                ax.fill_between(x, Q1[:, i], Q3[:, i], 
                                alpha=0.3, color=color, label=f'{label}.CI95')
    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels,loc='lower right', ncol=3, draggable=True)

"""compare multi dim data function individually from multiple sources"""
def compare_multi_dim_data(x_list:list,data_list:list,
                           dim:int,labels:list,xtype:str,datatype:str,
                           split=4,shuffle=False,
                           sharex:bool=True,semilogx:bool=False,legend:bool=True,
                           fig_label:str="Take1",
                           show_stats:bool=False,show_zero_cross:bool=False,
                           prev_fig=None,prev_ax=None):
    colors = get_n_colors(len(x_list),split,shuffle)
    # init fig, axes
    if prev_fig is not None and prev_ax is not None:
        fig = prev_fig
        axs = prev_ax
    else:
        fig = plt.figure(figsize=(8, 5),num=fig_label,tight_layout=True)
        fig.canvas.manager.window.wm_geometry("+100+100")
        axs = []
    
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
            dim_labels = [f"{datatype}_{i}" for i in ['trunk_ie','trunk_aa','trunk flex / extension',
                            'clav_dep_ev','clav_prot_ret',
                            'shoulder_fe','shoulder_aa','shoulder_ie',
                            'elbow_fe','elbow_ps','wrist_fe','wrist_dev']]
        else:
            dim_labels = [f"${datatype}_{{{i}}}$ (Nm)" for i in [r'trunk\,int-ext\,rot',r'trunk\,abd-adduct',r'trunk\,flex-extension',
                            r'clav\,dep-elev',r'clav\,prot-ret',
                            r'shoulder\,flex-extension',r'shoulder\,abd-adduct',r'shoulder\,int-ext\,rot',
                            r'elbow\,flex-extension',r'elbow\,pro-supination']]
        for i in range(dim):
            if dim >= 3:
                axs.append(fig.add_subplot(int(np.ceil(dim/3)),3,i+1))
            else:
                axs.append(fig.add_subplot(1,dim,i+1))
            axs[i].set_xlabel(f'{xtype}',fontsize=14)
            axs[i].set_ylabel(f'{dim_labels[i]}',fontsize=14)
            # axs[i].set_title(f'{dim_labels[i]} vs {xtype}')
            axs[i].grid(True)
            
    # reset the limits
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
        if len(data_list) > 0:
            if datatype == "q":
                axs[i].set_ylim(limits[i][0],limits[i][1])
            elif datatype == "q_rad":
                axs[i].set_ylim(np.deg2rad(limits[i][0]),np.deg2rad(limits[i][1]))
            else:
                axs[i].set_ylim(limits[0]-0.1*abs(limits[0]),limits[1]+0.1*abs(limits[1]))
    
    # plot actual data
    for i,(x_array_i,data_array_i,label,color) in enumerate(zip(x_list, data_list, labels, colors)):
        if sharex:
            x_array_i = np.column_stack([x_array_i for i in range(dim)])

        if data_array_i.shape[1] != dim:
            data_array_i = np.column_stack([data_array_i for i in range(dim)])
            
        if "Recon" in label:
            linewidth = 3
        else:
            linewidth = 2
        alpha = 1
        for j,ax in enumerate(axs):
            if semilogx:
                ax.semilogx(x_array_i[:, j], data_array_i[:, j], ls='-',color=color, label=label, alpha=alpha, linewidth=linewidth)
            else:
                ax.plot(x_array_i[:, j], data_array_i[:, j], ls='-',color=color, label=label, alpha=alpha, linewidth=linewidth)
    
    # plot stats
    if show_stats:
        plot_stats(x_list[0],[data_list], fig=fig,axs=axs,labels=["data"],legend=False,dist="gaussian")

    # sanity check for zero crossings cuz why not
    for j,ax in enumerate(axs):
        if show_zero_cross:
            plot_velocity_zero_crossings(x_array_i[:, j], data_array_i[:, j], ax)
    
    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels,loc='lower right', ncol=3, draggable=True)

    plt.tight_layout()
    return fig,axs

""" hide / show plot interactively (click to hide/show) """
def interactive_plot(fig, axs):
    handles, labels = axs[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower right', ncol=3, draggable=True)
    
    # Map legend handles to original artists across all axes
    lined = {}
    for leghandle, label in zip(leg.legend_handles, labels):
        sample_id = label.split(".")
        leghandle.set_picker(5)  # Works for both Line2D and Patch objects
        lined[leghandle] = {
            "art":[],
            "var_id":sample_id[0],
            "label":label
            }
        for ax in axs:
            # Check lines
            for line in ax.get_lines():
                if line.get_label() == label:
                    lined[leghandle]["art"].append(line)
            # Check patches (e.g., bar plots)
            for patch in ax.patches:
                if patch.get_label() == label:
                    lined[leghandle]["art"].append(patch)
            # Check collections (e.g., fill_between creates PolyCollection)
            for coll in ax.collections:
                if coll.get_label() == label:
                    lined[leghandle]["art"].append(coll)

    
    def on_pick(event):
        legline = event.artist
        if legline in lined:
            label = lined[legline]["label"]
            var_id = lined[legline]["var_id"]
            origlines = lined[legline]["art"]
            vis = not origlines[0].get_visible() if origlines else True
            if "median" not in label:
                # hide/show individual plots
                for origline in origlines:
                    origline.set_visible(vis)
                legline.set_alpha(1.0 if vis else 0)
                fig.canvas.draw()
            else:
                #hide/show all items for that variations if pressed mean or CI
                for (leghandle,leghandle_dict) in lined.items():
                    if leghandle_dict["var_id"] == var_id:
                        origlines = leghandle_dict["art"]
                        for origline in origlines:
                            origline.set_visible(vis)
                        leghandle.set_alpha(1.0 if vis else 0)
                fig.canvas.draw()
    fig.canvas.mpl_connect('pick_event', on_pick)
        
    def on_click(event):
        # Right-click (button 3) to hide all
        if event.button == 3:
            for leghandle, origlines in lined.items():
                for origline in origlines["art"]:
                    origline.set_visible(False)
                leghandle.set_alpha(0)
            fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    def on_resize(event):
        fig.tight_layout()
        fig.canvas.draw()
    fig.canvas.mpl_connect('resize_event', on_resize)
    return leg

"""plot 3d traj function"""
def plot_3d_trajectory(traj_list:list,label_list:list,fig=None,ax=None,label="3D Traj"):
    n_colors = len(traj_list)
    colors = get_n_colors(n_colors)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(15, 5), subplot_kw={'projection': '3d'},num=label,tight_layout=True)

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

""" plot submovements for a single 3d trajectory"""
def plot_3d_submovements(traj,sbmvmt_indices):
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw={'projection': '3d'},tight_layout=True)
    fig.canvas.manager.window.wm_geometry("+1000+100")
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

""" plot violin plots"""
def remove_outliers_iqr(data, multiplier=1.5):
    """
    Remove outliers using the IQR (Interquartile Range) method.
    
    Parameters:
    -----------
    data : array-like
        Input data
    multiplier : float
        IQR multiplier for outlier detection. Default is 1.5 (standard).
        - 1.5: Standard (removes extreme outliers)
        - 3.0: Conservative (removes only very extreme outliers)
        
    Returns:
    --------
    cleaned_data : array
        Data with outliers removed
    outliers : array
        The outlier values that were removed
    mask : boolean array
        Mask indicating which points are NOT outliers
    """
    data = np.array(data)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    mask = (data >= lower_bound) & (data <= upper_bound)
    cleaned_data = data[mask]
    outliers = data[~mask]
    
    return cleaned_data, outliers, mask

def plot_violins(
    data_list,
    axis_num,
    x_labels=None,
    title="Metric Distribution",
    axis_title=["Categories"],
    ylabel="Values",
    cut=2,
    bw_method='scott',
    show_mean=True,
    show_median=True,
    show_whiskers=True,
    show_points=True,
    violin_width=0.4,
    prev_fig=None,prev_axs=None
):
    """
    Create smooth violin plot with matplotlib using manual KDE and extended tails.
    
    Parameters:
    -----------
    data_list : list of lists, each element of the first list level is a list of categories to be plotted as their own violin plots
    x_labels : list, optional
        Custom labels for categories x-axis 
    colors : list or dict, optional
        Colors for each category. Can be list of colors or dict mapping categories to colors
    title : str
        Plot title
    axis_title : str
        X-axis label
    ylabel : str
        Y-axis label
    cut : float
        Distance past extreme data points to extend the KDE.
        Larger values = longer, smoother tails. Default is 2.
    bw_method : str or float
        Bandwidth estimation method for KDE. 'scott', 'silverman', or a scalar.
    show_mean : bool
        Show mean as a red diamond point
    show_median : bool
        Show median line inside violin
    show_whiskers : bool
        Show whiskers from min to max with quartile markers
    show_points : bool
        Show individual data points with jitter
    legend_labels : list, optional
        Custom legend labels. If None, uses category names
    violin_width : float
        Maximum width of violin (0-0.5). Default is 0.4
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """       
    # Set colors
    colors = get_n_colors(len(x_labels),1)
    
    # Init figure
    if prev_fig == None and prev_axs == None:
        fig = plt.figure(figsize=(30,10),num=title,tight_layout=True)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        axs = []
        for i in range(axis_num):
            if axis_num >= 3:
                axs.append(fig.add_subplot(int(np.ceil(axis_num/3)),3,i+1))
            else:
                axs.append(fig.add_subplot(1,axis_num,i+1))
    else:
        fig = prev_fig
        axs = prev_axs
    
    # Positions for each category
    positions = np.arange(1, len(x_labels) + 1)
    
    # remove outliers in all sets
    cleaned_set_data = []
    for set_data in data_list:
        cleaned_data_list = []
        for data in set_data:
            cleaned,_,_ = remove_outliers_iqr(data)
            cleaned_data_list.append(cleaned)
        cleaned_set_data.append(cleaned_data_list)
    # standardize the scale for every axis
    all_data = np.concatenate([arr for sublist in cleaned_set_data for arr in sublist])
    data_range = np.max(all_data) - np.min(all_data)
        
    for ax,set_data,ax_title in zip(axs,cleaned_set_data,axis_title):
        # Customize plot
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, rotation=0, ha='center')
        ax.set_title(ax_title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        # Add padding to y-axis
        ax.set_ylim(np.min(all_data) - data_range * 0.25, np.max(all_data) + data_range * 0.1)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        # plot the <1 boundary line
        if "Coverage" in title:
            ax.plot([positions[0]-0.5, positions[-1]+0.5], 
                   [50, 50], 
                   color='black',
                   linestyle=':',
                   linewidth=2, 
                   zorder=7)
        else:
            ax.plot([positions[0]-0.5, positions[-1]+0.5], 
                   [1, 1], 
                   color='black',
                   linestyle=':',
                   linewidth=2, 
                   zorder=7)
        
        # Draw each violin with manual KDE
        for i, (pos, data, color) in enumerate(zip(positions, set_data, colors)):
            # Calculate KDE
            kde = stats.gaussian_kde(data, bw_method=bw_method)
            
            # Get data range
            data_min, data_max = np.min(data), np.max(data)
            
            # Get bandwidth for extension
            bandwidth = np.sqrt(kde.covariance[0, 0])
            
            # Create extended evaluation points (smooth tails like seaborn)
            extend = cut * bandwidth
            y_eval = np.linspace(data_min - extend, data_max + extend, 200)
            
            # Evaluate KDE density
            density = kde(y_eval)
            
            # Normalize and scale density for width
            density = density / density.max() * violin_width
            
            # Plot violin (mirrored density for symmetric appearance)
            ax.fill_betweenx(
                y_eval, 
                pos - density, 
                pos + density,
                facecolor=color, 
                alpha=0.7, 
                edgecolor='black', 
                linewidth=1.5,
                zorder=2
            )
            
            # Add median line
            if show_median:
                median = np.median(data)
                ax.plot([pos - 0.04, pos + 0.04], 
                       [median, median], 
                       color='orange', 
                       linewidth=2, 
                       zorder=5)
        
            # Add points arranged as horizontal histogram
            if show_points:
                # Create bins along y-axis
                bin_edges = np.linspace(data_min - extend, data_max + extend, 26)
                
                # Assign each data point to a bin
                bin_indices = np.digitize(data, bin_edges)
                
                # Process each bin
                for bin_idx in range(1, len(bin_edges)):
                    # Get points in this bin
                    mask = bin_indices == bin_idx
                    y_in_bin = data[mask]
                    n_points = len(y_in_bin)
                    
                    if n_points > 0:
                        # Calculate bin center for y-position
                        bin_center = (bin_edges[bin_idx-1] + bin_edges[bin_idx]) / 2
                        
                        # Calculate width based on selected mode
                        # Use KDE density at this y-value
                        bin_density = kde(bin_center)[0]
                        max_density = kde(y_eval).max()
                        width_scale = (bin_density / max_density) * violin_width
                        
                        # Arrange points horizontally within the bin
                        if n_points == 1:
                            # Single point at center
                            x_positions = [pos]
                        else:
                            # Distribute points evenly across the width
                            x_positions = np.linspace(
                                pos - width_scale,
                                pos ,
                                n_points
                            )
                        
                        # Plot points
                        ax.scatter(x_positions, y_in_bin,
                                  alpha=0.3, 
                                  s=10,
                                  color=colors[i], 
                                  edgecolors='black', 
                                  linewidths=0.5,
                                  zorder=1)
                        
            # Add whiskers (quartiles and min/max)
            if show_whiskers:
                q1, q3 = np.percentile(data, [25, 75])
                whisker_min = np.min(data)
                whisker_max = np.max(data)
                
                # Main whisker line from min to max
                ax.plot([pos, pos], [whisker_min, whisker_max], 
                       color='black', linewidth=1.5, alpha=0.7, zorder=3,
                       solid_capstyle='round')
                
                # Min cap
                ax.plot([pos - 0.04, pos + 0.04], [whisker_min, whisker_min], 
                       color='black', linewidth=2.5, zorder=3,
                       solid_capstyle='round')
                
                # Max cap
                ax.plot([pos - 0.04, pos + 0.04], [whisker_max, whisker_max], 
                       color='black', linewidth=2.5, zorder=3,
                       solid_capstyle='round')
                
                # Quartile patch
                quartile_box = Rectangle(
                    xy=(pos - 0.04, q1),  # Bottom-left corner
                    width=0.08,              # Box width
                    height=q3 - q1,                        # Box height (Q3 - Q1)
                    facecolor='black',          # Fill color
                    alpha=1,              # Transparencys
                    zorder=4                               # Layer order
                )
                ax.add_patch(quartile_box)
    
            # Add mean points
            if show_mean:
                ax.scatter(pos, np.mean(data) , color='red', s=25, zorder=5, 
                      marker='D', edgecolors='black', linewidths=1.5)
    
    # Create legend
    legend_elements = []
    # legend_elements = [
    #     Patch(facecolor=colors[i], edgecolor='black', 
    #           label=x_labels[i], alpha=0.7) 
    #     for i in range(len(x_labels))
    # ]
    
    legend_elements.append(
        Line2D([0], [0], linestyle=':',linewidth=2, color='black', label='Perfomance Boundary')
    )
    if show_mean:
        legend_elements.append(
           Line2D([0], [0], marker='D', color='w', 
                             markerfacecolor='red', markersize=8,
                             markeredgecolor='black', label='Mean'))
    if show_whiskers:
        legend_elements.append(
            Line2D([0], [0], linestyle='-',linewidth=2, color='orange', label='Median')
        )
    fig.legend(handles=legend_elements,loc='upper right', ncol=3, draggable=True)
    
    # Tight layout
    plt.tight_layout()
    
    def on_resize(event):
        fig.tight_layout()
        fig.canvas.draw()
    fig.canvas.mpl_connect('resize_event', on_resize)
    
    fig.savefig(f'plots/{title}.png')
    return fig, ax