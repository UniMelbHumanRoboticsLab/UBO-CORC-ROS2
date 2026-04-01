import numpy as np

def compute_central_tendency(samples_arr):
    mean = np.mean(np.array(samples_arr),axis=0)
    max = np.max(np.array(samples_arr),axis=0)
    min = np.min(np.array(samples_arr),axis=0)
    median = np.median(np.array(samples_arr),axis=0)
    median = (max+min)/2
    
    return mean,median,max,min