
"""
upper torso class definition for upper torso model handling and inverse kinematics

@author: JQ
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys,os

# Set NumPy print options to show decimal format instead of exponential
np.set_printoptions(
    precision=3,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__)))
from ub_models import *

class ub():
    def __init__(self, body_params={'torso':0.5,'clav':0.2,'ua_l': 0.3, 'fa_l': 0.25, 'ha_l': 0.1, 'm_ua': 2.0, 'm_fa':1.1+0.23+0.6,"shoulder_aa_offset":[16],"ft_offsets": [0.1,0.1,0.1]},model='xsens',arm_side="right"):

        self.model = model
        if self.model == 'ubo':
            self.ub_model = ubo_robot(
                torso=body_params['torso'],
                clav=body_params['clav'],
                ua_l=body_params['ua_l'],
                fa_l=body_params['fa_l'],
                ha_l=body_params['ha_l'],
                m_ua=body_params['m_ua'],
                m_fa=body_params['m_fa'],
                shoulder_aa_offset=body_params['shoulder_aa_offset'],
                ft_offsets =body_params['ft_offsets'],
                arm_side=arm_side
            )
            self.ee_names = ["hand","clavicle","upper_arm","forearm"]
        elif self.model == 'xsens':
            self.ub_model = [xsens_ub_12dof(
                torso=body_params['torso'],
                clav=body_params['clav'],
                ua_l=body_params['ua_l'],
                fa_l=body_params['fa_l'],
                ha_l=body_params['ha_l'],
                m_ua=body_params['m_ua'],
                m_fa=body_params['m_fa'],
                shoulder_aa_offset=body_params['shoulder_aa_offset'],
                arm_side=arm_side
            )]
            self.ee_names = ["hand"]
        # dominant hand
        self.cur_side = arm_side
        
        #Overall arm mass
        self.Marm=0
        for l in self.ub_model[0].links:
            self.Marm+=l.m

        #Default gravity vector (can be changed)
        self.ub_model[0].gravity=[0,0,-9.81]
        
    """
    Estimate new joint angles (task parameters) from new task points
    - TODO: develop an IK that can recreate that without giving the explicit joint angles 
    """
    def IK_task_params(self,task_points,joints_traj,point_index,visual_ik):
        qq_new = []
        qq_new.append(joints_traj.values[point_index[0]]) # initial configuration
        for i,task_point in enumerate(task_points):
            TT=SE3(task_point)
            iter = 0
            success=0
            while (success == 0 and iter < 3):
                iter = iter+1
                # To replace with a custom IK that follows healthy body constraints
                q0 = np.array(joints_traj.values[point_index[i+1]].tolist())
                qq, success, iterations, searches, residual=self.ub_model[0].ik_GN(TT, q0=q0, mask=[1, 1, 1, 1, 1, 1],tol=1e-6,pinv=True)
            qq_new.append(qq)
            if success == 0:
                assert 0
            print(f"Success / Iterations/ Searches:",success,iterations,searches)
            print("Intended Task Param:",q0*180/np.pi,"\nCalculated Task Param:",qq*180/np.pi)
            print("Actual Task Point:",task_point.t)
            print("Assumed Task Point:",self.ub_model[0].fkine(q0).t)
            print("Estimated Task Point:",self.ub_model[0].fkine(qq).t)
            print()

        qq_new.append(joints_traj.values[point_index[-1]-1]) # take the one just before resting submovement
        if visual_ik:
            fig = plt.figure()
            self.ub_model[0].plot(q = np.array(qq_new),
                        backend='pyplot',block=False,loop=False,jointaxes=True,eeframe=True,shadow=False,fig=fig,dt=3)
            for qq in qq_new:
                joints_pose=self.ub_model[0].fkine_all(np.array(qq)) # this has all the frames of the robot joints
                ee_pose = [self.ub_model[0].fkine(np.array(qq))]
                self.plot_joints_ee_frames(joints_pose,ee_pose)
            
        qq_new = pd.DataFrame(qq_new, columns=list(joints_traj.columns))
        return qq_new
        
    def ArmMassFromBodyMass(self, body_mass: float):
        '''Calculate arm mass from overall body mass based on anthropomorphic rules
        from Drillis et al., Body Segment Parameters, 1964. Table 7'''
        UA_percent_m = 0.053
        FA_percent_m = 0.036
        hand_percent_m = 0.013
        self.ub_model[0][2].m = UA_percent_m*body_mass
        self.ub_model[0][4].m = (FA_percent_m+hand_percent_m)*body_mass
        self.Marm=0
        for l in self.ub_model[0].links:
            self.Marm+=l.m

    def SetGravity(self,g_vector: np.array =[0,0,-9.81]):
        '''Define (set) gravitational vector of the model'''
        self.ub_model[0].gravity=g_vector

    def ub_fkine(self,ub_posture):
        joints_pose = self.ub_model[0].fkine_all(np.deg2rad(np.array(ub_posture))) # this has all the frames of the robot joints

        ee_pose = []
        for i, robot in enumerate(self.ub_model): 
            posture = ub_posture[:(robot.n)] 
            ee_pose.append(robot.fkine(np.deg2rad(np.array(posture))))

            # print(robot)
            # if i == robot.n-1:
            #     block = True
            # else:
            #     block = False
            # robot.plot(np.deg2rad(np.array(posture)),block=block,fig=plt.figure(len(plt.get_fignums())+1))

        return joints_pose, ee_pose
    
    def plot_joints_ee_frames(self,joints_pose,ee_poses,show_joint_frames=True,show_ee_frames=True):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(-0.5,0.5)
        # ax.set_box_aspect([1, 1, 1])

        # plot skeleton
        x = np.array([joints_pose[0].t[0],joints_pose[4].t[0],joints_pose[6].t[0],joints_pose[8].t[0],joints_pose[10].t[0]])
        y = np.array([joints_pose[0].t[1],joints_pose[4].t[1],joints_pose[6].t[1],joints_pose[8].t[1],joints_pose[10].t[1]])
        z = np.array([joints_pose[0].t[2],joints_pose[4].t[2],joints_pose[6].t[2],joints_pose[8].t[2],joints_pose[10].t[2]])
        ax.plot(x,y,z,marker='o', linestyle='-', color='k')

        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())+list(mcolors.TABLEAU_COLORS.values())
        if show_joint_frames:
            for i,robot_joint_pose in enumerate(joints_pose):
                if i == 0 :
                    off = np.array([0.0,0.0,0.0])
                else:   
                    off = np.array([0.02,-0.02,-0.02])
                offset = SE3.Rt(robot_joint_pose.R, robot_joint_pose.t+robot_joint_pose.R@off)
                offset.plot(frame=f"{i}", length=0.05, ax=ax,color=colors[i],flo=(0.005,0.005,0.005))
        if show_ee_frames:
            for i, (ee_pose, frame_name) in enumerate(zip(ee_poses,self.ee_names)):
                ee_pose.plot(frame=frame_name, length=0.05, ax=ax,color='k',flo=(0.01,0.01,0.01))

if __name__ == "__main__":
    #Define ISB rtb arm model
    body_params = {'torso':0.5,
                    'clav': 20/100,
                    'ua_l': 34/100,
                    'fa_l': 28/100,
                    'ha_l': 0.05,
                    'm_ua': 2.0,
                    'm_fa': 1.1+0.23+0.6,
                    "shoulder_aa_offset": [17,10],
                    "ft_offsets": [0.2,0.2,0.2]}
    ub_xsens = ub(body_params,model="ubo",arm_side="right")
    ub_xsens_left = ub(body_params,model="ubo",arm_side="left")

    """
    Sanity check your UA model here
    """
    ub_postures_to_test = [[0,0,0,0,0,72,35,0,0,90,0,0]]

    for ub_posture in tqdm(ub_postures_to_test):
        joints_pose, ee_pose = ub_xsens.ub_fkine(ub_posture)
        ub_xsens.plot_joints_ee_frames(joints_pose,ee_pose)

        # joints_pose, ee_pose = ub_xsens_left.ub_fkine(ub_posture)
        # ub_xsens_left.plot_joints_ee_frames(joints_pose,ee_pose)
        plt.show(block=True)
        
        