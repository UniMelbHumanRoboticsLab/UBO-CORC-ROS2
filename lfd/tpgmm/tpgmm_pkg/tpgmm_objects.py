import numpy as np

"""
Class for each Task Params
"""
class TaskParams:
    def __init__(self, A, b, invA, num_of_gauss,num_of_frames,num_of_dim):
        self.A = A
        self.b = b
        self.invA = invA
        self.Mu = np.zeros(shape=(num_of_frames,num_of_gauss,num_of_dim,1))
        self.Sigma = np.zeros(shape=(num_of_frames,num_of_gauss,num_of_dim,num_of_dim))

"""
Class for each sample
"""
class Sample:
    def __init__(self, params:TaskParams, Data, nbData):
        self.params = params
        self.Data = Data
        self.nbData = nbData