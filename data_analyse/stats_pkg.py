import numpy as np

def compute_central_tendency(samples_arr):
    mean = np.mean(np.array(samples_arr),axis=0)
    max = np.max(np.array(samples_arr),axis=0)
    min = np.min(np.array(samples_arr),axis=0)
    median = np.median(np.array(samples_arr),axis=0)
    median = (max+min)/2

    std = np.std(np.array(samples_arr), axis=0, ddof=0)  # ddof=1 for sample std
    n = np.array(samples_arr).shape[0]  # number of repetitions
    # 95% CI using t-distribution
    sem = std / np.sqrt(n)  # standard error of mean
    from scipy import stats
    t_crit = stats.t.ppf(0.975, df=n-1) 
    ci_95 = t_crit * sem  # ✅ Margin of error           
    u = mean + ci_95
    l = mean - ci_95
    return mean,median,max,min,u,l

def remove_outliers_iqr(ori_data, multiplier=1.5):
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
    if len(ori_data.shape) == 1:
        data = np.expand_dims(ori_data,axis=1)
    else:
        data = ori_data
        
    outliers = []
    masks = []
    for dim in range(data.shape[1]):
        cur_dim_data = data[:,dim]
        
        q1 = np.percentile(cur_dim_data, 25)
        q3 = np.percentile(cur_dim_data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        mask = (cur_dim_data >= lower_bound) & (cur_dim_data <= upper_bound)
        masks.append(mask)
        outlier = data[~mask]
        outliers.append(outlier)
    total_mask = np.array(masks).T
    final_mask = np.all(total_mask, axis=1)
    cleaned_data = data[final_mask,:]

    return cleaned_data, outliers, mask