import numpy as np
import cupy as cp
from tqdm import tqdm

realmin  = np.finfo(np.float64).tiny

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from tpgmm_util import *
from tpgmm_objects import TaskParams

CONST2PI = 2*np.pi
"""
Class for CuPy Implementation TPGMM
"""
class TPGMM_cupy:
    def __init__(self, num_of_gauss, num_of_frames, num_of_dim,priors,kP,kV):
        self.num_of_gauss = num_of_gauss
        self.num_of_frames = num_of_frames
        self.num_of_dim = num_of_dim
        self.priors = priors
        self.kP = kP
        self.kV = kV

    """
    initialize the GMM parameters for each Gaussian in each frame
    """
    def init_gmm (self,tp_trajs):
        diagRegularizationFactor = 1E-4 #Optional regularization term
        tp_trajs = cp.asarray(np.reshape(tp_trajs.T,(tp_trajs.shape[1]*tp_trajs.shape[2],tp_trajs.shape[0])))
        sampledTime = tp_trajs[0,:]
        TimingSep =cp.linspace(np.min(sampledTime),np.max(sampledTime),self.num_of_gauss+1)
        Mu = cp.zeros(shape=(self.num_of_frames*self.num_of_dim, self.num_of_gauss))
        Sigma = cp.zeros(shape=(self.num_of_frames*self.num_of_dim, self.num_of_frames*self.num_of_dim, self.num_of_gauss))

        # initialize the mean and covariance matrices within each time window
        for i in range(self.num_of_gauss):
            window = cp.where(cp.logical_and(sampledTime>=TimingSep[i],sampledTime<TimingSep[i+1]) == True)[0]
            Mu[:,i] = cp.mean(tp_trajs[:,window],1)
            Sigma[:,:,i] = cp.cov(tp_trajs[:,window]) + cp.eye(tp_trajs.shape[0])*diagRegularizationFactor
            self.priors[i] = window.shape[0]
        self.priors = cp.asarray(self.priors / sum(self.priors)); 

        # reshape the mean and sigma array (num_of_dim,num_of_frames,num_of_gauss)
        self.Mu = cp.zeros((self.num_of_dim,self.num_of_frames,self.num_of_gauss))
        self.Sigma = cp.zeros(shape=(self.num_of_dim,self.num_of_dim,self.num_of_frames,self.num_of_gauss))
        for i in range(self.num_of_gauss):
            self.Mu[:,:,i] = cp.reshape(Mu[:,i],(self.num_of_frames,self.num_of_dim)).T
            for j in range(self.num_of_frames):
                sigmaCurGaussCurFrame = Sigma[j*self.num_of_dim:(j+1)*self.num_of_dim,j*self.num_of_dim:(j+1)*self.num_of_dim,i]
                self.Sigma[:,:,j,i] = sigmaCurGaussCurFrame

    """
    perform EM to get the most optimal mean and covariance for each Gaussian in each frame
    """
    def gaussPDF(self,Data,Mu,Sigma):
        """
        gaussian probability distribution function  
        """
        num_of_dim = Data.shape[0] # this is because fit and reproduce has different uses for this gaussPDF

        Data = Data - cp.expand_dims(Mu,axis=-1)
        prob = cp.sum(cp.matmul(cp.linalg.inv(Sigma),Data)*Data,axis=0)
        prob = cp.exp(-0.5*prob) / cp.sqrt(cp.power(CONST2PI,num_of_dim) * cp.abs(cp.linalg.det(Sigma)) + realmin)
        return prob

    def computeGamma(self,tp_trajs):
        
        nbData = tp_trajs.shape[0]
        Lik = cp.ones((self.num_of_gauss, nbData))
        GAMMA0 = cp.zeros((self.num_of_gauss, self.num_of_frames, nbData))
        for i in range(self.num_of_gauss):
            for j in range(self.num_of_frames):
                data_mat = tp_trajs[:,:,j].T
                GAMMA0[i,j,:] = self.gaussPDF(data_mat, self.Mu[:,j,i], self.Sigma[:,:,j,i])                
                Lik[i,:] = cp.multiply(Lik[i,:],(GAMMA0[i,j,:]))
            Lik[i,:] = Lik[i,:] * self.priors[i]
        GAMMA = Lik / cp.sum(Lik,0)+realmin
        return Lik, GAMMA, GAMMA0

    def fit_em(self,nbMinSteps,nbMaxSteps,maxDiffLL,diagRegFact,updateComp,tp_trajs):
        tp_trajs_cupy = cp.asarray(tp_trajs)
        maxDiffLL = cp.asarray(maxDiffLL)
        """
        Fit the data into the TPGMM model

        nbMinSteps:min number of allowed iterations
        nbMaxSteps:max number of allowed iterations
        maxDiffL:Likelihood increase threshold to stop the algorithm
        diagRegFact:optional regularization
        updateComp:flag to update prior,sigma and mu
        """    
        nbData = tp_trajs_cupy.shape[0]
        prevLL = cp.asarray(0)

        for iter in tqdm(range(nbMaxSteps)):
            # E-step
            L, GAMMA, GAMMA0 = self.computeGamma(tp_trajs_cupy)
            GAMMA2 = GAMMA / cp.expand_dims(cp.sum(GAMMA,1),axis=-1)
            self.Pix = GAMMA2

            # M-step
            for i in range(self.num_of_gauss):
                # Update Priors
                if updateComp[0]:
                    self.priors[i] = cp.sum(cp.sum(GAMMA[i,:])) / nbData

                for j in range(self.num_of_frames):
                    data_mat = tp_trajs_cupy[:,:,j].T
                    # Update Mu
                    if updateComp[1]:
                        self.Mu[:,j,i] = cp.squeeze(cp.matmul(data_mat,cp.expand_dims(GAMMA2[i,:].T,axis=-1)))

                    # update sigma
                    if updateComp[2]:
                        DataTmp = data_mat - cp.expand_dims(self.Mu[:,j,i],axis=-1)
                        self.Sigma[:,:,j,i] = cp.matmul(DataTmp,cp.matmul(cp.diag(GAMMA2[i,:]),DataTmp.T)) + cp.eye(DataTmp.shape[0]) * diagRegFact
            
            # compute Average Log-likelihood to estimate convergence
            curLL = cp.sum(cp.log(cp.sum(L,0))) / L.shape[1]
            if iter>nbMinSteps:
                if (curLL-prevLL)<maxDiffLL or iter==nbMaxSteps:
                    print(f"\n{curLL}")
                    print(prevLL)
                    print(f'EM converged after {iter+1} iterations.')
                    return True
            prevLL = curLL
        return True
    
    """
    reproduce an optimal trajectory based on the transformed Gaussians using new task parameters
    """
    def dynSysControl(self,DataIn,expected_data,sampleParam,last_input_index,last_output_index,new_dt):
        nbData = DataIn.shape[0]
        nbOut = last_output_index-last_input_index
        X = sampleParam.b[0,last_input_index+1:last_output_index+1]
        dX = cp.zeros((nbOut,1))
        L = cp.hstack((cp.eye(nbOut)*self.kP, cp.eye(nbOut)*self.kV))
        Input_reconOutput = cp.zeros((self.num_of_dim,nbData))
        for t in range(nbData):
            ddx = -cp.matmul(L,cp.vstack((X-cp.expand_dims(expected_data[:,t],axis=-1),dX)))
            # update the velocity with the corrected acceleration
            dX = dX + ddx * new_dt
            X = X + dX * new_dt
            if len(DataIn[t,:].shape) == 1: # to standardize for multi-input cases
                DataInTmp = cp.expand_dims(DataIn[t,:],axis=-1)
            Input_reconOutput[:,t] = cp.squeeze(cp.vstack((DataInTmp,X)),axis=-1)
        expectedOutput = Input_reconOutput[last_input_index+1:last_output_index+1,:]
        return Input_reconOutput,expectedOutput # return
    
    def conditionTPGMM(self,sampleParam):

        """
        Recondition the TPGMM using product of linearly transformed Gaussian
        """
        Sigma = cp.zeros((self.num_of_dim,self.num_of_dim,self.num_of_gauss))
        Mu = cp.zeros((self.num_of_dim,self.num_of_gauss))
        for i in range(self.num_of_gauss):
            sigmaTmp = cp.zeros((self.num_of_dim,self.num_of_dim))
            MuTmp = cp.zeros((self.num_of_dim,1))
            for j in range(self.num_of_frames):
                muCurFrame = cp.expand_dims(cp.matmul(sampleParam.A[j],self.Mu[:,j,i]),axis=-1) + sampleParam.b[j]
                sigmaCurFrame = cp.matmul(sampleParam.A[j],cp.matmul(self.Sigma[:,:,j,i],sampleParam.A[j].T))
                sigmaTmp = sigmaTmp + cp.linalg.inv(sigmaCurFrame)
                MuTmp = MuTmp + cp.matmul(cp.linalg.inv(sigmaCurFrame),muCurFrame)
            Sigma[:,:,i] = cp.linalg.inv(sigmaTmp)
            Mu[:,i] = cp.squeeze(cp.matmul(Sigma[:,:,i],MuTmp))
        return Mu, Sigma
    
    def GMR(self,tpNewMu,tpNewSigma,DataIn,last_input_index,last_output_index):
        """
        GMR to produce a reference trajectory to follow
        """
        nbData = DataIn.shape[0]
        nbOut = last_output_index-last_input_index
        diagRegFact = 1E-8; #Regularization term is optional
        muTmp = cp.zeros((nbOut,self.num_of_gauss))
        expected_data = cp.zeros((nbOut,nbData))
        expSigma = cp.zeros((nbOut,nbOut,nbData))
        priorsOutput = cp.zeros((self.num_of_gauss,nbData))
        for t in range(nbData):
            # compute overall activation weights from the each Gaussian
            for j in range(self.num_of_gauss):
                priorsOutput[j,t] = self.priors[j]*self.gaussPDF(Data=DataIn[t,:],Mu=tpNewMu[0:last_input_index+1,j],Sigma=tpNewSigma[0:last_input_index+1,0:last_input_index+1,j])
            priorsOutput[:,t] = priorsOutput[:,t] / cp.sum(priorsOutput[:,t]+realmin) # normalize activation weights
            
            for j in range(self.num_of_gauss):
                # Compute conditional means from each Gaussian and sum it to get the expected output mean 
                muTmp[:,j] = tpNewMu[last_input_index+1:last_output_index+1,j] + cp.matmul(cp.matmul(tpNewSigma[last_input_index+1:last_output_index+1,0:last_input_index+1,j],cp.linalg.inv(tpNewSigma[0:last_input_index+1,0:last_input_index+1,j])),DataIn[t,:]-tpNewMu[0:last_input_index+1,j])
                expected_data[:,t] = expected_data[:,t] + priorsOutput[j,t]*muTmp[:,j]
            
                SigmaTmp = tpNewSigma[last_input_index+1:last_output_index+1,last_input_index+1:last_output_index+1,j] - cp.matmul(cp.matmul(tpNewSigma[last_input_index+1:last_output_index+1,0:last_input_index+1,j],cp.linalg.inv(tpNewSigma[0:last_input_index+1,0:last_input_index+1,j])),tpNewSigma[0:last_input_index+1,last_input_index+1:last_output_index+1,j])
                expSigma[:,:,t] = expSigma[:,:,t] + cp.multiply(priorsOutput[j,t],SigmaTmp+cp.matmul(muTmp[:,j],muTmp[:,j].T))
            expSigma[:,:,t] = expSigma[:,:,t] - cp.matmul(expected_data[:,t],expected_data[:,t].T) + cp.eye(nbOut,nbOut)*diagRegFact
        return expected_data,expSigma,priorsOutput

    def reproduce(self,DataIn,sampleParam,last_input_index,last_output_index,DS,new_dt):

        # convert everything to CuPy objects
        tp_cupy = TaskParams(cp.asarray(sampleParam.params.A),cp.asarray(sampleParam.params.b),
                             cp.asarray(sampleParam.params.invA),0,0,0)
        tp_cupy.Mu = cp.asarray(sampleParam.params.Mu)
        tp_cupy.Sigma = cp.asarray(sampleParam.params.Sigma)
        DataIn = cp.asarray(DataIn)
        if len(DataIn.shape) == 1: # to standardize for multi-input cases
            DataIn = cp.expand_dims(DataIn,axis=-1)

        tpNewMu,tpNewSigma = self.conditionTPGMM(tp_cupy)
        expected_data,expSigma,priorsOutput = self.GMR(tpNewMu,tpNewSigma,DataIn,last_input_index,last_output_index)
        if DS: 
            _,expected_dyn_data = self.dynSysControl(DataIn,expected_data,tp_cupy,last_input_index,last_output_index,new_dt)
        else:
            expected_dyn_data = None
        return expected_data.get(),expSigma,priorsOutput,expected_dyn_data.get()

        



    
    