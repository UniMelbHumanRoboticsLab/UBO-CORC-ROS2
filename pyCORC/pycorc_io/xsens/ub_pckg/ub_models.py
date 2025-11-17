# -*- coding: utf-8 -*-
"""
torso model functions that defines a rbt-toolbox robot

supported models
- 12 DOF upper torso model based on XSENS Biomechanical model
- 5 DOF upper limb model based on ISB recommendations using Vive trackers

@author: JQ
"""
import numpy as np
from spatialmath import SO3, SE3
import roboticstoolbox as rbt
#convenience function returning unit vector from a to b
def unit(from_a, to_b):
    return (to_b-from_a)/np.linalg.norm(to_b-from_a)
#convenience function returning unit vector of vec
def unitV(vec):
    return vec/np.linalg.norm(vec)

############### All Available Upper Limb Models ##############################
def xsens_ub_12dof(torso:float,clav:float,ua_l: float, fa_l: float, ha_l: float, m_ua: float = 0, m_fa: float = 0,arm_side:str = "right") -> rbt.Robot:
    """
    xsens_ub_12dof Create a Robot of robotic toolbox Xsens compatible upper body (pelvis to hand)
    torso: torso length
    clav: clavicle length
    ua_l: upper-arm length
    fa_l: forearm length
    ha_l: hand length
    m_ua and m_fa: upper-arm and forearm masses, centered in middle of segment
    - internal/external rotation are not following ISBUL standard for rehabilitation application
    """

    if arm_side == "right":
        L = [] #Links list
        # ROM: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7549223/#:~:text=Normal%20range%20of%20active%20movement,for%20external%20rotation%20%5B6%5D.
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=0,offset=np.pi/2,name='trunk_ie'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi/2,name='trunk_aa'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=0,name='trunk_fe'))
        L.append(rbt.RevoluteMDH(d=0,a=torso,alpha=np.pi/2,offset=np.pi/2,name='scapula_de'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=-np.pi/2,name='scapula_pr'))
        L.append(rbt.RevoluteMDH(d=clav,a=0,alpha=np.pi/2,offset=-np.pi/2,name='shoulder_fe'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=-np.pi/2,name='shoulder_aa'))
        L.append(rbt.RevoluteMDH(d=-ua_l,a=0,alpha=np.pi/2,offset=-np.pi/2,name='shoulder_ie'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi,name='elbow_fe'))
        L.append(rbt.RevoluteMDH(d=-fa_l,a=0,alpha=np.pi/2,offset=np.pi,name='elbow_ps'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi/2,name='wrist_fe'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi/2,name='wrist_dev'))

        xsens = rbt.DHRobot(L)

        #Add hand transformation (tool) to match XSENS model wrist offset
        xsens.base=SE3(SO3.Rz(np.pi/2))
        xsens.tool=SE3(SO3.Ry(0))*SE3(SO3.Rx(0))*SE3([0,ha_l,0]) # for intrinsic rotation (rotation about local axis), always use post multiply
    elif arm_side == "left":
        L = [] #Links list
        # ROM: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7549223/#:~:text=Normal%20range%20of%20active%20movement,for%20external%20rotation%20%5B6%5D.
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=0,offset=np.pi/2,name='trunk_ie'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi/2,name='trunk_aa'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi,name='trunk_fe'))
        L.append(rbt.RevoluteMDH(d=0,a=-torso,alpha=np.pi/2,offset=np.pi/2,name='scapula_de'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=-np.pi/2,name='scapula_pr'))
        L.append(rbt.RevoluteMDH(d=-clav,a=0,alpha=np.pi/2,offset=-np.pi/2,name='shoulder_fe'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=-np.pi/2,name='shoulder_aa'))
        L.append(rbt.RevoluteMDH(d=ua_l,a=0,alpha=np.pi/2,offset=-np.pi/2,name='shoulder_ie'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi,name='elbow_fe'))
        L.append(rbt.RevoluteMDH(d=fa_l,a=0,alpha=np.pi/2,offset=np.pi,name='elbow_ps'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=np.pi/2,name='wrist_fe'))
        L.append(rbt.RevoluteMDH(d=0,a=0,alpha=np.pi/2,offset=-np.pi/2,name='wrist_dev'))

        xsens = rbt.DHRobot(L)

        # #Add hand transformation (tool) to match XSENS model wrist offset
        xsens.base=SE3(SO3.Rz(np.pi/2))
        xsens.tool=SE3(SO3.Ry(np.pi))*SE3(SO3.Rx(0))*SE3([0,ha_l,0]) # for intrinsic rotation (rotation about local axis), always use post multiply
    return xsens

def vive_ub_5dof(ua_l: float, fa_l: float, ha_l: float, m_ua: float = 0, m_fa: float = 0,arm_side: str="right") -> rbt.Robot:
    """
    Create a Robot of robotic toolbox engineering compatible arm w/ shoulder elbow and wrist
    ua_l: upper-arm length
    fa_l: forearm length
    ha_l: hand length
    m_ua and m_fa: upper-arm and forearm masses, centered in middle of segment
    right_dom: dominant hand is left or right
    # singularity occurs when elevation is zero, lose a degree of freesom
    """
    if arm_side == "right":
        L = [] #Links list
        # ROM: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7549223/#:~:text=Normal%20range%20of%20active%20movement,for%20external%20rotation%20%5B6%5D.
        L.append(rbt.RevoluteMDH(d=0,       a=0.0,  alpha=np.pi/2,  offset=-np.pi/2, qlim=[-90/180*np.pi, 90/180*np.pi],                                  name='shoulder_fe')) # shoulder flex/extend
        L.append(rbt.RevoluteMDH(d=0,       a=0.0,  alpha=np.pi/2,  offset=np.pi/2,  qlim=[-45/180*np.pi, 135/180*np.pi],                                  name='shoulder_aa')) # shoulder abduct/adduct
        L.append(rbt.RevoluteMDH(d=-ua_l,   a=0.0,  alpha=-np.pi/2, offset=np.pi/2,  qlim=[-90/180*np.pi, 105/180*np.pi],                                  name='shoulder_ie')) # upper arm/shoulder Int/ext
        L.append(rbt.RevoluteMDH(d=0,       a=0,    alpha=np.pi/2,  offset=-np.pi/2, qlim=[-90/180*np.pi, 105/180*np.pi],                                  name='elbow_fe')) # Elbow flex
        L.append(rbt.RevoluteMDH(d=-fa_l,   a=0.0,  alpha=np.pi/2,  offset=-np.pi/2, qlim=[-90/180*np.pi, 120/180*np.pi],     m = m_fa, r = [0,0,-fa_l/2], name='elbow_ps')) # Pronosupination

        ISBUL = rbt.DHRobot(L)

        #Add hand transformation (tool) to match OpenSIM model wrist offset
        #frame: z -> x, x -> -y, y -> -z
        ISBUL.base=SE3(SO3.Rz(np.pi/2))
        ISBUL.tool=SE3(SO3.Rx(np.pi/2))*SE3(SO3.Rz(-np.pi))*SE3([0,ha_l,0])
        # ISBUL.tool=SE3([0,ha_l,0])
    elif arm_side == "left":
        L = [] #Links list
        # ROM: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7549223/#:~:text=Normal%20range%20of%20active%20movement,for%20external%20rotation%20%5B6%5D.
        L.append(rbt.RevoluteMDH(d=0,       a=0.0,  alpha=np.pi/2,  offset=np.pi/2, qlim=[-90/180*np.pi, 90/180*np.pi],                                  name='shoulder_fe')) # shoulder flex/extend
        L.append(rbt.RevoluteMDH(d=0,       a=0.0,  alpha=np.pi/2,  offset=-np.pi/2,  qlim=[-45/180*np.pi, 135/180*np.pi],                                  name='shoulder_aa')) # shoulder abduct/adduct
        L.append(rbt.RevoluteMDH(d=ua_l,    a=0.0,  alpha=np.pi/2,  offset=-np.pi/2,  qlim=[-90/180*np.pi, 105/180*np.pi],                                  name='shoulder_ie')) # upper arm/shoulder Int/ext
        L.append(rbt.RevoluteMDH(d=0,       a=0,    alpha=np.pi/2,  offset=-np.pi/2, qlim=[-90/180*np.pi, 105/180*np.pi],                                  name='elbow_fe')) # Elbow flex
        L.append(rbt.RevoluteMDH(d=fa_l,    a=0.0,  alpha=np.pi/2,  offset=-np.pi/2, qlim=[-90/180*np.pi, 120/180*np.pi],     m = m_fa, r = [0,0,-fa_l/2], name='elbow_ps')) # Pronosupination

        ISBUL = rbt.DHRobot(L)

        #Add hand transformation (tool) to match OpenSIM model wrist offset
        #frame: z -> x, x -> -y, y -> -z
        ISBUL.base=SE3(SO3.Rz(np.pi/2))
        ISBUL.tool=SE3([0,ha_l,0])
        ISBUL.tool=SE3(SO3.Rx(-np.pi/2))*SE3(SO3.Rz(-np.pi))*SE3([0,ha_l,0])
    return ISBUL