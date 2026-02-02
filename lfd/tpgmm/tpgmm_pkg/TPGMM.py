import numpy as np
np.set_printoptions(suppress=True,precision=4) # suppress scientific notation
from tqdm import tqdm
realmin  = np.finfo(np.float64).tiny
realmax  = np.finfo(np.float64).max

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from tpgmm_util import *
from tpgmm_objects import Sample,TaskParams

"""
Class for TPGMM
"""
""" get max norm diff between 2 n dimensional arrays"""
def get_max_norm_diff(arr1,arr2):
    diff = np.abs(arr1 - arr2)
    max_index = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    max_norm_diff = diff[max_index]/(np.abs(arr2[max_index])+realmin)
    return max_norm_diff

class TPGMM:
    def __init__(self, num_of_gauss, num_of_frames, num_of_dim,priors,kP,kV,diagRegFact,version="fast",init_type='kmeans++'):
        self.num_of_gauss = num_of_gauss
        self.num_of_frames = num_of_frames
        self.num_of_dim = num_of_dim
        self.priors = priors
        self.kP = kP
        self.kV = kV
        self.diagRegFact = diagRegFact
        self.version = version
        self.init_type = init_type
        print(f"Initialized TPGMM with {num_of_gauss} Gaussians, {num_of_frames} frames, {num_of_dim} dimensions, version: {version}, init_type: {init_type}")

    """
    initialize the GMM parameters for each Gaussian in each frame
    """
    def init_gmm (self,tp_data):
        # time window init
        if self.init_type == 'time':
            tp_data = np.reshape(tp_data.T,(tp_data.shape[1]*tp_data.shape[2],tp_data.shape[0]))

            Mu = np.zeros(shape=(self.num_of_frames*self.num_of_dim, self.num_of_gauss))
            Sigma = np.zeros(shape=(self.num_of_frames*self.num_of_dim, self.num_of_frames*self.num_of_dim, self.num_of_gauss))
            sampledTime = tp_data[0,:]
            TimingSep = np.linspace(np.min(sampledTime),np.max(sampledTime),self.num_of_gauss+1)
            # initialize the mean and covariance matrices within each time window
            for i in range(self.num_of_gauss):
                window = np.where(np.logical_and(sampledTime>=TimingSep[i],sampledTime<TimingSep[i+1]) == True)[0]
                Mu[:,i] = np.mean(tp_data[:,window],1)
                Sigma[:,:,i] = np.cov(tp_data[:,window]) + np.eye(tp_data.shape[0])*self.diagRegFact
                self.priors[i] = window.shape[0]
            self.priors = self.priors / sum(self.priors); 
            # reshape the mean and sigma array (num_of_dim,num_of_frames,num_of_gauss)
            self.Mu = np.zeros((self.num_of_dim,self.num_of_frames,self.num_of_gauss))
            self.Sigma = np.zeros(shape=(self.num_of_dim,self.num_of_dim,self.num_of_frames,self.num_of_gauss))
            for i in range(self.num_of_gauss):
                self.Mu[:,:,i] = np.reshape(Mu[:,i],(self.num_of_frames,self.num_of_dim)).T
                for j in range(self.num_of_frames):
                    sigmaCurGaussCurFrame = Sigma[j*self.num_of_dim:(j+1)*self.num_of_dim,j*self.num_of_dim:(j+1)*self.num_of_dim,i]
                    self.Sigma[:,:,j,i] = sigmaCurGaussCurFrame
        # Kmeans++ init
        else:
            from gmr.gmm import kmeansplusplus_initialization
            from gmr.gmm import covariance_initialization
            # start with equal priors for each gaussian
            self.priors = np.random.dirichlet(np.ones(self.num_of_gauss))
            # self.priors = self.priors / self.num_of_gauss
            self.Mu = np.zeros((self.num_of_dim,self.num_of_frames,self.num_of_gauss))
            self.Sigma = np.zeros(shape=(self.num_of_dim,self.num_of_dim,self.num_of_frames,self.num_of_gauss))
            
            for i in range(self.num_of_frames):
                dataCurFrame = tp_data[:,:,i]
                # initialize the mean for current frame
                MuCurFrame = kmeansplusplus_initialization(
                    dataCurFrame, self.num_of_gauss
                )
                
                # initialize the covariance for current frame
                SigmaCurFrame = covariance_initialization(
                    dataCurFrame,self.num_of_gauss)

                for j in range(self.num_of_gauss):
                    self.Mu[:,i,j] = MuCurFrame[j]
                    self.Sigma[:,:,i,j] = SigmaCurFrame[j] + np.eye(self.num_of_dim)*self.diagRegFact
            print(f"Init Priors:\t{self.priors}\tSum:{np.sum(self.priors)}\n")

    """
    perform EM to get the most optimal mean and covariance for each Gaussian in each frame
    """
    def gaussPDF(self,Data,Mu,Sigma):
        """
        gaussian probability distribution function  
        """
        num_of_dim = Data.shape[0] # this is because fit and reproduce has different uses for this gaussPDF

        if Data.ndim == 1:
            Data =  np.expand_dims(Data,axis=-1)
        if Mu.ndim == 1:
            Mu = np.expand_dims(Mu,axis=-1)
        
        Data = Data - Mu
        prob = sum(np.matmul(np.linalg.inv(Sigma),Data)*Data,0)
        prob = np.exp(-0.5*prob) / (np.sqrt(np.power(2*np.pi,num_of_dim) * np.abs(np.linalg.det(Sigma))))
        return prob
    def computeGamma_slow(self,tp_data):
        nbData = tp_data.shape[0]

        # ============ Original version ============
        # init responsibility of each main gaussian for each data point
        Lik = np.ones((self.num_of_gauss, nbData))

        # init responsibility of each gaussian in each frame for each data point
        GAMMA0 = np.zeros((self.num_of_gauss, self.num_of_frames, nbData))
        
        for i in range(self.num_of_gauss):
            for j in range(self.num_of_frames):
                data_mat = tp_data[:,:,j].T
                GAMMA0[i,j,:] = self.gaussPDF(data_mat, self.Mu[:,j,i], self.Sigma[:,:,j,i])                
                Lik[i,:] = np.multiply(Lik[i,:],(GAMMA0[i,j,:]))
            Lik[i,:] = Lik[i,:] * self.priors[i]
        Lik = np.clip(Lik, realmin, realmax) # to prevent numerical issues
        GAMMA = Lik / np.sum(Lik,0)

        return Lik, GAMMA, GAMMA0
    def computeGamma(self,tp_data):
        # ============ Vectorized version ============
        # tp_data: (nbData, num_of_dim, num_of_frames)
        # self.Mu: (num_of_dim, num_of_frames, num_of_gauss)
        # self.Sigma: (num_of_dim, num_of_dim, num_of_frames, num_of_gauss)

        # Expand tp_data for broadcasting: (nbData, num_of_dim, num_of_frames, 1)
        data_expanded = tp_data[:, :, :, np.newaxis]

        # Expand Mu for broadcasting: (1, num_of_dim, num_of_frames, num_of_gauss)
        Mu_expanded = self.Mu[np.newaxis, :, :, :]

        # Compute difference: (nbData, num_of_dim, num_of_frames, num_of_gauss)
        diff = data_expanded - Mu_expanded

        # Transpose Sigma for batch inversion: (num_of_frames, num_of_gauss, num_of_dim, num_of_dim)
        Sigma_transposed = np.transpose(self.Sigma, (2, 3, 0, 1))
        inv_Sigma = np.linalg.inv(Sigma_transposed)

        # Transpose diff to (num_of_frames, num_of_gauss, num_of_dim, nbData)
        diff_transposed = np.transpose(diff, (2, 3, 1, 0))

        # Compute quadratic form: diff.T @ inv_Sigma @ diff for each (frame, gauss, data)
        # inv_Sigma @ diff: (f, g, d, n)
        inv_sigma_diff = np.einsum('fgij,fgjn->fgin', inv_Sigma, diff_transposed)
        # Sum over dim: (f, g, n)
        quad = np.einsum('fgin,fgin->fgn', inv_sigma_diff, diff_transposed)

        # Compute determinants: (num_of_frames, num_of_gauss)
        dets = np.linalg.det(Sigma_transposed)

        # Normalization constant: (num_of_frames, num_of_gauss)
        norm = np.sqrt((2 * np.pi) ** self.num_of_dim * np.abs(dets))

        # Compute PDF: (num_of_frames, num_of_gauss, nbData)
        pdf = np.exp(-0.5 * quad) / norm[:, :, np.newaxis]

        # GAMMA02: (num_of_gauss, num_of_frames, nbData)
        GAMMA0 = np.transpose(pdf, (1, 0, 2))

        # Lik2: product over frames, multiplied by priors
        Lik_unclipped = np.prod(GAMMA0, axis=1) * self.priors[:, np.newaxis]
        Lik = np.clip(Lik_unclipped, realmin, realmax) # to prevent numerical issues

        # GAMMA2: normalized responsibilities
        GAMMA = Lik / np.sum(Lik, axis=0)
        
        return Lik, GAMMA, GAMMA0
    def fit_em(self,nbMinSteps,nbMaxSteps,maxDiffLL,updateComp,tp_data):
        """
        Fit the data into the TPGMM model

        nbMinSteps:min number of allowed iterations
        nbMaxSteps:max number of allowed iterations
        maxDiffL:Likelihood increase threshold to stop the algorithm
        updateComp:flag to update prior,sigma and mu
        """    
        nbData = tp_data.shape[0]
        LL = []
        pbar = tqdm(range(nbMaxSteps), ncols=120)

        for iter in pbar:
            # Fast EM
            if self.version=="fast" or self.version=="compare":
                # E-step (vectorized)
                L, GAMMA, GAMMA0 = self.computeGamma(tp_data)
                GAMMA_SUM = GAMMA / np.expand_dims(np.sum(GAMMA,1),axis=-1)
                self.Pix = GAMMA_SUM

                # M-step (vectorized)
                priors = self.priors
                Mu = self.Mu
                Sigma = self.Sigma
                # Update Priors: (num_of_gauss,)
                if updateComp[0]:
                    priors = np.sum(GAMMA, axis=1) / nbData
                # Update Mu: (num_of_dim, num_of_frames, num_of_gauss)
                # tp_data: (nbData, num_of_dim, num_of_frames), GAMMA2: (num_of_gauss, nbData)
                if updateComp[1]:
                    Mu = np.einsum('ndf,gn->dfg', tp_data, GAMMA_SUM)
                else:
                    Mu = self.Mu.copy()
                # Update Sigma: (num_of_dim, num_of_dim, num_of_frames, num_of_gauss)
                if updateComp[2]:
                    # Compute difference using the UPDATED Mu: (nbData, num_of_dim, num_of_frames, num_of_gauss)
                    diff = tp_data[:, :, :, np.newaxis] - Mu[np.newaxis, :, :, :]
                    # Weighted outer product summed over data points
                    # diff shape: (n, d, f, g), GAMMA2 shape: (g, n)
                    # Output Sigma shape: (d, d, f, g)
                    Sigma = np.einsum('gn,nafg,nbfg->abfg', GAMMA_SUM, diff, diff) + np.eye(self.num_of_dim)[:, :, np.newaxis, np.newaxis] * self.diagRegFact # add regularization to prevent singular matrix
                # compute Average Log-likelihood to estimate convergence
                curLL = np.sum(np.log(np.sum(L,0))) / L.shape[1]
            # Slow EM
            if self.version=="slow" or self.version=="compare":
                # E-step slow
                L2,GAMMA2,GAMMA02 = self.computeGamma_slow(tp_data)
                GAMMA_SUM2 = GAMMA2 / np.expand_dims(np.sum(GAMMA2,1),axis=-1)
                self.Pix = GAMMA_SUM2

                # M-step slow
                priors2 = self.priors
                Mu2 = self.Mu
                Sigma2 = self.Sigma
                for i in range(self.num_of_gauss):
                    # Update Priors
                    if updateComp[0]:
                        priors2[i] = np.sum(np.sum(GAMMA2[i,:])) / nbData
                    for j in range(self.num_of_frames):
                        data_mat = tp_data[:,:,j].T
                        # Update Mu
                        if updateComp[1]:
                            Mu2[:,j,i] = np.squeeze(np.matmul(data_mat,np.expand_dims(GAMMA_SUM2[i,:].T,axis=-1)))
                        # update sigma
                        if updateComp[2]:
                            DataTmp = data_mat - np.expand_dims(Mu2[:,j,i],axis=-1)
                            Sigma2[:,:,j,i] = np.matmul(DataTmp,np.matmul(np.diag(GAMMA_SUM2[i,:]),DataTmp.T)) + np.eye(DataTmp.shape[0]) * self.diagRegFact # add regularization to prevent singular matrix
                # compute Average Log-likelihood to estimate convergence
                curLL = np.sum(np.log(np.sum(L2,0))) / L2.shape[1]
            # assign updated parameters
            if self.version=="fast":
                self.priors = priors
                self.Mu = Mu
                self.Sigma = Sigma
            elif self.version=="slow" or self.version=="compare":
                self.priors = priors2
                self.Mu = Mu2
                self.Sigma = Sigma2
            # Verify vectorized version matches original
            if self.version=="compare":
                print(f"\niter:{iter}")
                print("Max norm diff Lik:", get_max_norm_diff(L,L2))
                print("Max norm diff GAMMA:", get_max_norm_diff(GAMMA,GAMMA2))
                print("Max norm diff GAMMA0:", get_max_norm_diff(GAMMA0,GAMMA02))
                print("Max norm diff GAMMA_SUM2:", get_max_norm_diff(GAMMA_SUM,GAMMA_SUM2))
                print("Max norm diff Priors:", get_max_norm_diff(priors,priors2))
                print("Max norm diff Mu:", get_max_norm_diff(Mu,Mu2))
                print("Max norm diff Sigma:", get_max_norm_diff(Sigma,Sigma2))
                print()

            LL.append(curLL)
            diff = np.abs((LL[iter]-LL[iter-1]))
            pbar.set_postfix(iter=iter,cur_ll=curLL,loglike_diff=diff)

            if iter>nbMinSteps:
                if diff<maxDiffLL or  iter==nbMaxSteps-1:
                    break

        print(f"Likelihood {iter}: {LL[iter]:.4f} \t Tol: {diff:.4f}")
        print(f"Conv. Priors:\t{self.priors}\tSum:{np.sum(self.priors)}")
        print(f'EM converged after {iter+1} iterations.')
        return np.array([LL]).T
    
    """
    reproduce optimal trajectory by conditioning GMM with task parameters then GMR (with slow and fast implementation)
    """
    def dynSysControl(self,DataIn,expected_data,sampleParam:TaskParams,last_input_index,last_output_index,new_dt):
        if len(DataIn.shape) == 1: # to standardize for multi-input cases
            DataIn = np.expand_dims(DataIn,axis=-1)
        nbData = DataIn.shape[0]
        nbOut = last_output_index-last_input_index
        X = sampleParam.params.b[0,last_input_index+1:last_output_index+1]
        dX = np.zeros((nbOut,1))
        L = np.hstack((np.eye(nbOut)*self.kP, np.eye(nbOut)*self.kV))
        Input_reconOutput = np.zeros((self.num_of_dim,nbData))
        for t in range(nbData):
            np.vstack((X-np.expand_dims(expected_data[:,t],axis=-1),dX))
            ddx = -np.matmul(L,np.vstack((X-np.expand_dims(expected_data[:,t],axis=-1),dX)))
            # update the velocity with the corrected acceleration
            dX = dX + ddx * new_dt
            X = X + dX * new_dt
            if len(DataIn[t,:].shape) == 1: # to standardize for multi-input cases
                DataInTmp = np.expand_dims(DataIn[t,:],axis=-1)
            Input_reconOutput[:,t] = np.squeeze(np.vstack((DataInTmp,X)),axis=-1)
        expectedOutput = Input_reconOutput[last_input_index+1:last_output_index+1,:]
        return Input_reconOutput,expectedOutput # return
    def conditionTPGMM_slow(self,sampleParam):
        """
        Recondition the TPGMM using product of linearly transformed Gaussian
        """
        # Slow version for verification
        Sigma = np.zeros((self.num_of_dim,self.num_of_dim,self.num_of_gauss))
        Mu = np.zeros((self.num_of_dim,self.num_of_gauss))
        for i in range(self.num_of_gauss):
            sigmaTmp = np.zeros((self.num_of_dim,self.num_of_dim))
            MuTmp = np.zeros((self.num_of_dim,1))
            for j in range(self.num_of_frames):
                muCurFrame = np.expand_dims(np.matmul(sampleParam.A[j],self.Mu[:,j,i]),axis=-1) + sampleParam.b[j]
                sigmaCurFrame = np.matmul(sampleParam.A[j],np.matmul(self.Sigma[:,:,j,i],sampleParam.A[j].T))
                sigmaTmp = sigmaTmp + np.linalg.inv(sigmaCurFrame)
                MuTmp = MuTmp + np.matmul(np.linalg.inv(sigmaCurFrame),muCurFrame)
            Sigma[:,:,i] = np.linalg.inv(sigmaTmp)
            Mu[:,i] = np.squeeze(np.matmul(Sigma[:,:,i],MuTmp))
        return Mu, Sigma
    def GMR_slow(self,tpNewMu,tpNewSigma,DataIn,last_input_index,last_output_index):
        """
        GMR to produce a reference trajectory to follow
        """
        if len(DataIn.shape) == 1: # to standardize for multi-input cases
            DataIn = np.expand_dims(DataIn,axis=-1)
        nbData = DataIn.shape[0]
        nbOut = last_output_index-last_input_index
        muTmp = np.zeros((nbOut,self.num_of_gauss))
        expected_data = np.zeros((nbOut,nbData))
        expSigma = np.zeros((nbOut,nbOut,nbData))
        priorsOutput = np.zeros((self.num_of_gauss,nbData))
        for t in range(nbData):
            # compute overall activation weights from the each Gaussian
            for j in range(self.num_of_gauss):
                # print(t,j)
                p = self.gaussPDF(Data=DataIn[t,:],Mu=tpNewMu[0:last_input_index+1,j],Sigma=tpNewSigma[0:last_input_index+1,0:last_input_index+1,j])
                priorsOutput[j,t] = self.priors[j]*p
            priorsOutput[:,t] = np.clip(priorsOutput[:,t], realmin, realmax) # to prevent numerical issues
            priorsOutput[:,t] = priorsOutput[:,t] / np.sum(priorsOutput[:,t]) # normalize activation weights
            
            for j in range(self.num_of_gauss):
                # Compute conditional means from each Gaussian and sum it to get the expected output mean 
                muTmp[:,j] = tpNewMu[last_input_index+1:last_output_index+1,j] + np.matmul(np.matmul(tpNewSigma[last_input_index+1:last_output_index+1,0:last_input_index+1,j],np.linalg.inv(tpNewSigma[0:last_input_index+1,0:last_input_index+1,j])),DataIn[t,:]-tpNewMu[0:last_input_index+1,j])
                expected_data[:,t] = expected_data[:,t] + np.multiply(priorsOutput[j,t],muTmp[:,j])
            
                # compute conditional covariance from each Gaussian and condition it to expected output covariance
                SigmaTmp = tpNewSigma[last_input_index+1:last_output_index+1,last_input_index+1:last_output_index+1,j] - np.matmul(np.matmul(tpNewSigma[last_input_index+1:last_output_index+1,0:last_input_index+1,j],np.linalg.inv(tpNewSigma[0:last_input_index+1,0:last_input_index+1,j])),tpNewSigma[0:last_input_index+1,last_input_index+1:last_output_index+1,j])
                expSigma[:,:,t] = expSigma[:,:,t] + np.multiply(priorsOutput[j,t],SigmaTmp+np.matmul(muTmp[:,j],muTmp[:,j].T))
            expSigma[:,:,t] = expSigma[:,:,t] - np.matmul(expected_data[:,t],expected_data[:,t].T) + np.eye(nbOut,nbOut)*self.diagRegFact
    
        return expected_data,expSigma,priorsOutput
    def conditionTPGMM(self,sampleParam:TaskParams):
        """
        Recondition the TPGMM using product of linearly transformed Gaussian
        """
        # Fast version
        # Stack A and b from sampleParam: A shape (f, d, d), b shape (f, d, 1)
        A = np.array([sampleParam.A[j] for j in range(self.num_of_frames)])  # (f, d, d)
        b = np.array([sampleParam.b[j] for j in range(self.num_of_frames)])  # (f, d, 1)
        
        # self.Mu: (d, f, g), self.Sigma: (d, d, f, g)
        
        # Transform Mu for all frames and Gaussians: muCurFrame = A @ Mu + b
        # Mu transposed to (f, d, g) for einsum
        Mu_t = np.transpose(self.Mu, (1, 0, 2))  # (f, d, g)
        # A @ Mu for each frame: (f, d, d) @ (f, d, g) -> (f, d, g)
        muCurFrame = np.einsum('fij,fjg->fig', A, Mu_t) + b  # (f, d, g)
        
        # Transform Sigma for all frames and Gaussians: sigmaCurFrame = A @ Sigma @ A.T
        # Sigma transposed to (f, g, d, d) for batch operations
        Sigma_t = np.transpose(self.Sigma, (2, 3, 0, 1))  # (f, g, d, d)
        # A @ Sigma @ A.T: (f, d, d) @ (f, g, d, d) @ (f, d, d).T -> (f, g, d, d)
        sigmaCurFrame = np.einsum('fij,fgjk,flk->fgil', A, Sigma_t, A)  # (f, g, d, d)
        
        # Invert sigmaCurFrame for all frames and Gaussians
        inv_sigmaCurFrame = np.linalg.inv(sigmaCurFrame)  # (f, g, d, d)
        
        # Sum inverse covariances over frames: sigmaTmp = sum_j inv(sigmaCurFrame[j])
        sigmaTmp = np.sum(inv_sigmaCurFrame, axis=0)  # (g, d, d)
        
        # Compute MuTmp = sum_j inv(sigmaCurFrame[j]) @ muCurFrame[j]
        # muCurFrame is (f, d, g), transpose to (f, g, d) for matrix multiply
        muCurFrame_t = np.transpose(muCurFrame, (0, 2, 1))  # (f, g, d)
        # inv_sigmaCurFrame @ muCurFrame and sum over frames
        MuTmp = np.einsum('fgij,fgj->gi', inv_sigmaCurFrame, muCurFrame_t)  # (g, d)
        
        # Final Sigma = inv(sigmaTmp)
        Sigma = np.linalg.inv(sigmaTmp)  # (g, d, d)
        
        # Final Mu = Sigma @ MuTmp
        Mu = np.einsum('gij,gj->gi', Sigma, MuTmp)  # (g, d)
        
        # Transpose to match original output shapes: Sigma (d, d, g), Mu (d, g)
        Sigma = np.transpose(Sigma, (1, 2, 0))  # (d, d, g)
        Mu = Mu.T  # (d, g) 
        return Mu, Sigma
    def GMR(self, tpNewMu, tpNewSigma, DataIn, last_input_index, last_output_index):
        """
        GMR to produce a reference trajectory to follow (vectorized over Gaussians)
        """
        if len(DataIn.shape) == 1:
            DataIn = np.expand_dims(DataIn, axis=-1)
        nbData = DataIn.shape[0]
        nbOut = last_output_index - last_input_index
        nbIn = last_input_index + 1
        
        expected_data = np.zeros((nbOut, nbData))
        expSigma = np.zeros((nbOut, nbOut, nbData))
        priorsOutput = np.zeros((self.num_of_gauss, nbData))
        
        # Precompute slices for all Gaussians
        # MuIn: (nbIn, g), MuOut: (nbOut, g)
        MuIn = tpNewMu[0:last_input_index+1, :]  # (nbIn, g)
        MuOut = tpNewMu[last_input_index+1:last_output_index+1, :]  # (nbOut, g)
        
        # SigmaIn: (nbIn, nbIn, g), SigmaOut: (nbOut, nbOut, g)
        SigmaIn = tpNewSigma[0:last_input_index+1, 0:last_input_index+1, :]  # (nbIn, nbIn, g)
        SigmaOut = tpNewSigma[last_input_index+1:last_output_index+1, last_input_index+1:last_output_index+1, :]  # (nbOut, nbOut, g)
        
        # Cross-covariances
        SigmaOutIn = tpNewSigma[last_input_index+1:last_output_index+1, 0:last_input_index+1, :]  # (nbOut, nbIn, g)
        SigmaInOut = tpNewSigma[0:last_input_index+1, last_input_index+1:last_output_index+1, :]  # (nbIn, nbOut, g)
        
        # Transpose to have g first for batch operations
        SigmaIn_t = np.transpose(SigmaIn, (2, 0, 1))  # (g, nbIn, nbIn)
        SigmaOut_t = np.transpose(SigmaOut, (2, 0, 1))  # (g, nbOut, nbOut)
        SigmaOutIn_t = np.transpose(SigmaOutIn, (2, 0, 1))  # (g, nbOut, nbIn)
        SigmaInOut_t = np.transpose(SigmaInOut, (2, 0, 1))  # (g, nbIn, nbOut)
        
        # Batch invert SigmaIn
        SigmaIn_inv = np.linalg.inv(SigmaIn_t)  # (g, nbIn, nbIn)
        
        # Precompute K = SigmaOutIn @ SigmaIn_inv for all Gaussians
        K = np.einsum('goj,gji->goi', SigmaOutIn_t, SigmaIn_inv)  # (g, nbOut, nbIn)
        
        # Precompute SigmaTmp_all = SigmaOut - K @ SigmaInOut for all Gaussians (time-independent)
        K_SigmaInOut = np.einsum('goj,gjp->gop', K, SigmaInOut_t)  # (g, nbOut, nbOut)
        SigmaTmp_all = SigmaOut_t - K_SigmaInOut  # (g, nbOut, nbOut)
        
        # For Gaussian PDF computation
        MuIn_t = MuIn.T  # (g, nbIn)
        detSigmaIn = np.linalg.det(SigmaIn_t)  # (g,)
        norm = np.sqrt((2 * np.pi) ** nbIn * np.abs(detSigmaIn))  # (g,)
        
        for t in range(nbData):
            # ===== Vectorized activation weights (priorsOutput) =====
            diff_in = DataIn[t, :] - MuIn_t  # (g, nbIn)
            
            # Quadratic form: diff_in @ SigmaIn_inv @ diff_in.T for each g
            tmp = np.einsum('gi,gij->gj', diff_in, SigmaIn_inv)  # (g, nbIn)
            quad = np.einsum('gi,gi->g', tmp, diff_in)  # (g,)
            
            # Compute PDF and activation weights
            pdf_unclip = np.exp(-0.5 * quad) / norm  # (g,)
            weighted_pdf = self.priors * pdf_unclip  # (g,)
            priorsOutput[:, t] = np.clip(weighted_pdf, realmin, realmax) # prevent numerical issues
            priorsOutput[:, t] = priorsOutput[:, t] / (np.sum(priorsOutput[:, t]))
            
            # ===== Vectorized conditional means (muTmp) =====
            # muTmp = MuOut + K @ diff_in
            # K: (g, nbOut, nbIn), diff_in: (g, nbIn) -> (g, nbOut)
            muTmp_t = np.einsum('goj,gj->go', K, diff_in)  # (g, nbOut)
            muTmp = MuOut + muTmp_t.T  # (nbOut, g)
            
            # Weighted sum for expected_data
            expected_data[:, t] = np.einsum('g,og->o', priorsOutput[:, t], muTmp)  # (nbOut,)
            
            # ===== Vectorized conditional covariance (expSigma) =====
            # Note: np.matmul(muTmp[:,j], muTmp[:,j].T) for 1D arrays computes inner product (scalar)
            # So we compute inner product for each Gaussian
            muTmp_inner = np.einsum('og,og->g', muTmp, muTmp)  # (g,) - inner product per Gaussian
            
            # SigmaTmp_all: (g, nbOut, nbOut), transpose to (nbOut, nbOut, g)
            SigmaTmp_t = np.transpose(SigmaTmp_all, (1, 2, 0))  # (nbOut, nbOut, g)
            # Add scalar inner product (broadcasts to all elements)
            cov_term = SigmaTmp_t + muTmp_inner  # (nbOut, nbOut, g)
            
            # Weighted sum over Gaussians
            expSigma[:, :, t] = np.einsum('g,opg->op', priorsOutput[:, t], cov_term)
            
            # Subtract expected inner product (matching slow version's np.matmul behavior for 1D)
            expected_inner = np.dot(expected_data[:, t], expected_data[:, t])  # scalar
            expSigma[:, :, t] = expSigma[:, :, t] - expected_inner + np.eye(nbOut) * self.diagRegFact
    
        return expected_data, expSigma, priorsOutput
    def repro_condition_gmr(self,DataIn,sample:Sample,last_input_index,last_output_index,DS,new_dt):
        # Fast GMR
        if self.version=="fast" or self.version=="compare":
            tpNewMu,tpNewSigma = self.conditionTPGMM(sample.params) 
            expected_data,expSigma,priorsOutput = self.GMR(tpNewMu,tpNewSigma,DataIn,last_input_index,last_output_index)
        # Slow GMR
        if self.version=="slow" or self.version=="compare":
            tpNewMu2,tpNewSigma2 = self.conditionTPGMM_slow(sample.params)
            expected_data2,expSigma2,priorsOutput2 = self.GMR_slow(tpNewMu2,tpNewSigma2,DataIn,last_input_index,last_output_index)

        # Verify vectorized version matches original
        if self.version=="compare":
            print() 
            print("Max norm diff new Mu:", np.max(np.abs(tpNewMu - tpNewMu2)))
            print("Max norm diff new Sigma:", np.max(np.abs(tpNewSigma - tpNewSigma2)))
            print("Max norm diff expected Mu:", np.max(np.abs(expected_data2 - expected_data)),expected_data.shape)
            print("Max norm diff expected Sigma:", np.max(np.abs(expSigma2 - expSigma)),expSigma.shape)
            print("Max norm diff expected Priors:", np.max(np.abs(priorsOutput2 - priorsOutput)))
            print() 

        # Slow GMR
        if self.version=="slow" or self.version=="compare":
            expected_data = expected_data2
            expSigma = expSigma2
            priorsOutput = priorsOutput2

        if DS: 
            _,expected_dyn_data = self.dynSysControl(DataIn,expected_data,sample.params,last_input_index,last_output_index,new_dt)
        else:
            expected_dyn_data = None
        return expected_data.T,expSigma,priorsOutput,expected_dyn_data
    
    """
    reproduce optimal trajectory with GMR first then transforming the expected mean with Task Parameters (with slow and fast implementation)
    """
    def repro_gmr_condition(self,DataIn,sampleParam:Sample,last_input_index,last_output_index):
        nbData = DataIn.shape[0]
        if DataIn.ndim == 1:
            DataIn = DataIn[:,np.newaxis]
        DataIn = DataIn.T # reshape to (num_of_dim,nbData)
        num_dim_out = last_output_index - last_input_index

        # compute conditional probability for each frame
        MuGMR = np.zeros((num_dim_out,nbData,self.num_of_frames))
        SigmaGMR = np.zeros((num_dim_out,num_dim_out,nbData,self.num_of_frames))
        
        for j in range (self.num_of_frames):
            # Compute activation weights
            priors_output = np.zeros((self.num_of_gauss,nbData))
            for i in range (self.num_of_gauss):
                priors_output[i,:] = self.priors[i]*self.gaussPDF(Data=DataIn,
                                                      Mu=self.Mu[0:last_input_index+1,j,i],
                                                      Sigma=self.Sigma[0:last_input_index+1,0:last_input_index+1,j,i])
            priors_output = np.clip(priors_output, realmin, realmax) # to prevent numerical issues
            priors_output = priors_output / np.sum(priors_output,0)

            for t in range(nbData):
                # compute conditional mean
                for i in range(self.num_of_gauss):
                    mu_i = self.Mu[0:last_input_index+1,j,i]
                    mu_o = self.Mu[last_input_index+1:last_output_index+1,j,i]
                    sigma_oi = self.Sigma[last_input_index+1:last_output_index+1,0:last_input_index+1,j,i]
                    sigma_io = self.Sigma[0:last_input_index+1,last_input_index+1:last_output_index+1,j,i]
                    sigma_oo = self.Sigma[last_input_index+1:last_output_index+1,last_input_index+1:last_output_index+1,j,i]
                    sigma_ii = self.Sigma[0:last_input_index+1,0:last_input_index+1,j,i]

                    data = DataIn[:,t]-mu_i

                    mu_tmp = mu_o + np.matmul(np.matmul(sigma_oi,np.linalg.inv(sigma_ii)),data)
                    MuGMR[:,t,j] = MuGMR[:,t,j] + priors_output[i,t]*mu_tmp

                    sigma_tmp = sigma_oo-np.matmul(np.matmul(sigma_oi,np.linalg.inv(sigma_ii)),sigma_io)
                    SigmaGMR[:,:,t,j] = SigmaGMR[:,:,t,j] + priors_output[i,t]*(sigma_tmp + np.matmul(np.expand_dims(mu_tmp,axis=-1),np.expand_dims(mu_tmp,axis=-1).T))
                SigmaGMR[:,:,t,j] = SigmaGMR[:,:,t,j] - np.matmul(np.expand_dims(MuGMR[:,t,j],axis=-1),np.expand_dims(MuGMR[:,t,j],axis=-1).T) + np.eye(num_dim_out,num_dim_out)*self.diagRegFact

        MuTmp = np.zeros((num_dim_out, nbData, self.num_of_frames))
        SigmaTmp = np.zeros((num_dim_out, num_dim_out, nbData, self.num_of_frames))

        for j in range(self.num_of_frames):
            A = sampleParam.params.A[j,last_input_index+1:last_output_index+1,last_input_index+1:last_output_index+1]
            b = sampleParam.params.b[j,last_input_index+1:last_output_index+1]
            MuTmp[:,:,j] = np.matmul(A,MuGMR[:,:,j]) + b
            for t in range(nbData):
                SigmaTmp[:,:,t,j] = np.matmul(np.matmul(A,SigmaGMR[:,:,t,j]),A.T)

        expected_sigma = np.zeros((num_dim_out,num_dim_out,nbData))
        expected_data = np.zeros((num_dim_out,nbData))
        for t in range(nbData):
            SigmaP = np.zeros((num_dim_out,num_dim_out))
            MuP = np.zeros((num_dim_out))
            for j in range(self.num_of_frames):
                SigmaP = SigmaP + np.linalg.inv(SigmaTmp[:,:,t,j])
                MuP = MuP + np.matmul(np.linalg.inv(SigmaTmp[:,:,t,j]),MuTmp[:,t,j])
            expected_sigma[:,:,t] = np.linalg.inv(SigmaP)
            expected_data[:,t] = np.matmul(expected_sigma[:,:,t],MuP)
        
        return expected_data.T,expected_sigma