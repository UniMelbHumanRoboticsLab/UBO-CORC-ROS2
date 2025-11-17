
"""
upper torso class definition for upper torso model handling and inverse kinematics

@author: JQ
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys,os
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__)))
from ub_models import *

class ub():
    def __init__(self, body_params={'torso':0.5,'clav':0.2,'ua_l': 0.3, 'fa_l': 0.25, 'ha_l': 0.1, 'm_ua': 2.0, 'm_fa':1.1+0.23+0.6},model='xsens',arm_side=True):

        self.model = model
        if self.model == 'xsens':
            self.ub_model = xsens_ub_12dof(body_params['torso'],
                                     body_params['clav'],
                                     body_params['ua_l'],
                                     body_params['fa_l'],
                                     body_params['ha_l'],
                                     body_params['m_ua'],
                                     body_params['m_fa'],
                                     arm_side=arm_side)

        # dominant hand
        self.cur_side = arm_side
        
        #Overall arm mass
        self.Marm=0
        for l in self.ub_model.links:
            self.Marm+=l.m

        #Default gravity vector (can be changed)
        self.ub_model.gravity=[0,0,-9.81]
        
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
                qq, success, iterations, searches, residual=self.ub_model.ik_GN(TT, q0=q0, mask=[1, 1, 1, 1, 1, 1],tol=1e-6,pinv=True)
            qq_new.append(qq)
            if success == 0:
                assert 0
            print(f"Success / Iterations/ Searches:",success,iterations,searches)
            print("Intended Task Param:",q0*180/np.pi,"\nCalculated Task Param:",qq*180/np.pi)
            print("Actual Task Point:",task_point.t)
            print("Assumed Task Point:",self.ub_model.fkine(q0).t)
            print("Estimated Task Point:",self.ub_model.fkine(qq).t)
            print()

        qq_new.append(joints_traj.values[point_index[-1]-1]) # take the one just before resting submovement
        if visual_ik:
            fig = plt.figure()
            self.ub_model.plot(q = np.array(qq_new),
                        backend='pyplot',block=False,loop=False,jointaxes=True,eeframe=True,shadow=False,fig=fig,dt=3)
            for qq in qq_new:
                robot_joints=self.ub_model.fkine_all(np.array(qq)) # this has all the frames of the robot joints
                robot_ee = self.ub_model.fkine(np.array(qq))
                self.plot_joints_ee_frames(robot_joints,robot_ee)
            
        qq_new = pd.DataFrame(qq_new, columns=list(joints_traj.columns))
        return qq_new
    
    def plot_joints_ee_frames(self,robot_joints,robot_ee):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(-0.5,0.5)
        ax.set_box_aspect([1, 1, 1])

        x = np.array([robot_joints[0].t[0],robot_joints[4].t[0],robot_joints[6].t[0],robot_joints[8].t[0],robot_joints[10].t[0]])
        y = np.array([robot_joints[0].t[1],robot_joints[4].t[1],robot_joints[6].t[1],robot_joints[8].t[1],robot_joints[10].t[1]])
        z = np.array([robot_joints[0].t[2],robot_joints[4].t[2],robot_joints[6].t[2],robot_joints[8].t[2],robot_joints[10].t[2]])
        ax.plot(x,y,z,marker='o', linestyle='-', color='k')
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())+list(mcolors.TABLEAU_COLORS.values())

        for i,robot_joint_pose in enumerate(robot_joints):
            if i == 0 :
                off = np.array([0.0,0.0,0.0])
            else:   
                off = np.array([0.02,-0.02,-0.02])
            offset = SE3.Rt(robot_joint_pose.R, robot_joint_pose.t+robot_joint_pose.R@off)
            offset.plot(frame=f"{i}", length=0.05, ax=ax,color=colors[i],flo=(0.005,0.005,0.005))
        robot_ee.plot(frame=f"hand", length=0.05, ax=ax,color='k',flo=(0.01,0.01,0.01))
        
    def ArmMassFromBodyMass(self, body_mass: float):
        '''Calculate arm mass from overall body mass based on anthropomorphic rules
        from Drillis et al., Body Segment Parameters, 1964. Table 7'''
        UA_percent_m = 0.053
        FA_percent_m = 0.036
        hand_percent_m = 0.013
        self.ub_model[2].m = UA_percent_m*body_mass
        self.ub_model[4].m = (FA_percent_m+hand_percent_m)*body_mass
        self.Marm=0
        for l in self.ub_model.links:
            self.Marm+=l.m

    def SetGravity(self,g_vector: np.array =[0,0,-9.81]):
        '''Define (set) gravitational vector of the model'''
        self.ub_model.gravity=g_vector

    def ub_fkine(self,ub_posture):
        robot_joints=self.ub_model.fkine_all(np.deg2rad(np.array(ub_posture))) # this has all the frames of the robot joints
        robot_ee = self.ub_model.fkine(np.deg2rad(np.array(ub_posture)))

        return robot_joints, robot_ee

if __name__ == "__main__":
    #Define ISB rtb arm model
    body_params = {'torso':0.5,
                    'clav': 20/100,
                    'ua_l': 34/100,
                    'fa_l': 28/100,
                    'ha_l': 0.05,
                    'm_ua': 2.0,
                    'm_fa': 1.1+0.23+0.6}
    ub_xsens = ub(body_params,model="xsens",arm_side="right")
    ub_xsens_left = ub(body_params,model="xsens",arm_side="left")

    """
    Test FA and UA
    """
    ub_postures_to_test = [[6.04,-1.5,1.23,1.3,-6.3,115,56,24,85,17,-15,15]]
    ub_postures_to_test = [[0,0,0,0,0,0,0,0,0,0,0,0],]
    plot = True
    for ub_posture in tqdm(ub_postures_to_test):
        #Compute theoretical link positions for each posture
        robot_joints, robot_ee = ub_xsens.ub_fkine(ub_posture)
        ub_xsens.plot_joints_ee_frames(robot_joints,robot_ee)

        robot_joints, robot_ee = ub_xsens_left.ub_fkine(ub_posture)
        ub_xsens_left.plot_joints_ee_frames(robot_joints,robot_ee)
        plt.show(block=True)
        # ub_xsens.ub_model.plot(np.deg2rad(np.array(ub_posture)),block=True)
        