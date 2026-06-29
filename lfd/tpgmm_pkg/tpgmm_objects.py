
"""
Class for each Task Params
"""
class TaskParams:
    def __init__(self, A, b, invA):
        self.A = A
        self.b = b
        self.invA = invA

"""
Class for each sample
"""
class Sample:
    def __init__(self, params:TaskParams, Data, nbData):
        self.params = params
        self.Data = Data
        self.nbData = nbData