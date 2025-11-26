import os, sys,json
import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

from ubo_replayer import ubo_replayer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.pycorc_gui import *

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pycorc_io.xsens.ub_pckg.ub import ub

from PySide6 import QtWidgets
from PySide6.QtGui import QShortcut
from PySide6.QtCore import QThread,Qt,QMetaObject,Slot


NUM_RFT = 3
class ubo_replay_gui(pycorc_gui):
    def __init__(self,init_args):
        self.init_args = init_args
        self.corc_args = self.init_args["init_flags"]["corc"]
        self.xsens_args = self.init_args["init_flags"]["xsens"]
        self.gui_args = self.init_args["init_flags"]["gui"]
        self.replay_args = self.init_args["init_flags"]["replay"]

        self.num_closed_threads = 0
        self.num_opened_threads = 0

        super().__init__(freq=self.gui_args["freq"],gui_3d=self.gui_args["3d"])

        if self.gui_args["force"]:
            self.ubo_wrenches_live_stream = self.init_live_stream(num_columns=2)
            
        """
        Initilization
        """
        self.init_IOs()
        self.init_shortcuts()
        # init_debug()

    """
    Init Sensor Worker / Thread / GUI Helper Functions
    """
    def add_opened_threads(self):
        self.num_opened_threads += 1
        print(f"Threads Opened:{self.num_opened_threads}")
    def init_corc(self):
        # init response label
        self.corc_label = self.init_response_label(size=[525,100])
        # init corc live stream plot
        self.ubo_F_live_stream_plots = []
        self.ubo_M_live_stream_plots = []
        if self.gui_args["on"] and self.gui_args["force"]:
            for i in range(NUM_RFT):
                self.ubo_F_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.ubo_wrenches_live_stream,sensor_name= f"UBO_{i} Force",unit="N",dim=3))
                self.ubo_M_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.ubo_wrenches_live_stream,sensor_name= f"UBO_{i} Moment",unit="Nm",dim=3))
    def init_xsens(self):
        # init response label
        self.xsens_label = self.init_response_label(size=[300,200])
        # init xsens ub model
        body_params = {'torso':50/100,
                        'clav': 20/100,
                        'ua_l': 34/100,
                        'fa_l': 28/100,
                        'ha_l': 0.05,
                        'm_ua': 2.0,
                        'm_fa': 1.1+0.23+0.6,
                        "shoulder_aa_offset": [ 17,10],
                        "ft_offsets": [0.05,0.05,0.05]}
        
        self.skeleton = {
            "right":{},
            "left":{}
        }
        # init xsens skeleton for 3d gui
        if self.gui_args["on"] and self.gui_args["3d"]:
            for side,color in zip(["right","left"],["purple", "orange"]):
                self.skeleton[side]["ub_xsens"] = ub(body_params,model="ubo",arm_side=side)
                robot_joints, robot_ee = self.skeleton[side]["ub_xsens"].ub_fkine([0]*12)
                body = self.init_line(points=robot_joints.t,color=color)
                ees = [self.init_frame(pos=ee_pose.t,rot=ee_pose.R*0.1) for ee_pose in robot_ee]
                self.skeleton[side]["body"] = body
                self.skeleton[side]["ees"] = ees
                if self.corc_args["on"]:
                    if side == "right":
                        forces = [0] + [self.init_line(points=np.vstack([ee_pose.t, ee_pose.t + np.matmul(ee_pose.R,np.array([0,0,0.2]))]),color="pink") for ee_pose in robot_ee[1:]]
                    else:
                        forces = [0 for ee_pose in robot_ee]
                else:
                    forces = [0 for ee_pose in robot_ee]
                self.skeleton[side]["forces"] = forces

    def init_replayer(self):
        # init response label
        self.replayer_label = self.init_response_label(size=[250,150])
        self.replayer_thread = QThread()
        self.replayer_worker = ubo_replayer(init_args = self.init_args["init_flags"],
                                            take_num  = self.init_args["take_num"])
    def init_shortcuts(self):
        # to start / stop logging at button press
        if self.replay_args["on"]:
            # to start logging
            start_log = QShortcut("S", self)
            start_log.activated.connect(self.replayer_worker.start_worker)
            # to stop logging and close everything
            stop_log = QShortcut("C", self)
            stop_log.activated.connect(self.replayer_worker.stop_replayer)
            stop_log.activated.connect(self.gui_timer.stop)
            
        close = QShortcut("Q", self)
        close.activated.connect(self.gui_timer.stop)
        close.activated.connect(self.close_workers)
    def init_IOs(self):
        """
        Init IO thread and worker
        """
        if self.corc_args["on"]:
            self.init_corc()
        if self.xsens_args["on"]:
            self.init_xsens()
        if self.replay_args["on"]:
            self.init_replayer()
            self.replayer_response = {
            "print_text":"Idle\n",
            "replayer_fps":0.0
        }
        
        """
        Move sensor worker to thread 
        Connect intersensor signals
        Connect signals to sensor GUI callbacks
        Connect thread start to start sensor worker, update number of open threads
        Connect worker stop to stop thread at closeup
        Connect thread finished to close app
        """
        if self.replay_args["on"]:
            # move worker to thread
            self.replayer_worker.moveToThread(self.replayer_thread)
            # connect replayer to sensor gui
            self.replayer_worker.time_ready.connect(self.update_replayer,type=Qt.ConnectionType.QueuedConnection)
            # connect thread start to add opened threads  
            self.replayer_thread.started.connect(self.add_opened_threads)
            # connect replayer finished saving to close workers
            self.replayer_worker.finish_save.connect(self.close_workers)
            # connect worker stop to stop thread at closeup
            self.replayer_worker.stopped.connect(self.replayer_thread.exit)
            # connect thread finished to close app
            self.replayer_thread.finished.connect(self.close_app)

        """
        Start threads
        """
        if self.replay_args["on"]:
            self.replayer_thread.start()
            self.replayer_thread.setPriority(QThread.Priority.TimeCriticalPriority)
        
    """
    Update Sensor Variable Callbacks
    """
    @Slot(dict)
    def update_replayer(self,replayer_response):
        self.replayer_response = replayer_response
        if self.replayer_response["corc"]:
            self.corc_response = self.replayer_response["corc"]
        if self.replayer_response["xsens"]:
            self.xsens_response = self.replayer_response["xsens"]
        
    """
    Update Main GUI Helper Functions and Callback
    """
    def update_gui(self):
        super().update_gui()
        if self.gui_args["on"]:
            # update live stream time buffer
            if self.gui_args["force"]:
                self.update_live_stream_buffer(live_stream=self.ubo_wrenches_live_stream)
        if hasattr(self, 'corc_response'):
            # update rft info
            corc_data = self.corc_response["raw_data"]
            txt = ""
            for i in range(NUM_RFT):
                txt += f"UBO_{i} Force(N): {corc_data[1+i*6]:8.4f} {corc_data[2+i*6]:8.4f} {corc_data[3+i*6]:8.4f} | Moment(Nm): {corc_data[4+i*6]:8.4f} {corc_data[5+i*6]:8.4f} {corc_data[6+i*6]:8.4f}\n"
            self.update_response_label(self.corc_label,f"CORC Running Time:{corc_data[0]}s\n{txt}")
            if self.gui_args["on"] and self.gui_args["force"]:
                for i in range(NUM_RFT):
                    force_data = corc_data[1+i*6:1+i*6+6]
                    self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_F_live_stream_plots[i],force_data,dim=3)
                    self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_M_live_stream_plots[i],force_data[3:],dim=3)
        if hasattr(self, 'xsens_response'):
            # update rft info
            right_list = self.xsens_response["right"]
            left_list = self.xsens_response["left"]

            txt = ""
            for i,key in  enumerate(['trunk_ie','trunk_aa','trunk_fe',
                        'scapula_de','scapula_pr',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps','wrist_fe','wrist_dev']):
                txt += f"{key:15}: {left_list[i]:8.4f} {right_list[i]:8.4f}\n"

            if self.gui_args["on"] and self.gui_args["3d"]:
                for side in ["right","left"]:
                    ub_posture = self.xsens_response[side]
                    robot_joints, robot_ee = self.skeleton[side]["ub_xsens"].ub_fkine(ub_posture)
                    self.update_line(self.skeleton[side]["body"],points=robot_joints.t)
                    for i,(frame,force,pose) in enumerate(zip(self.skeleton[side]["ees"],self.skeleton[side]["forces"],robot_ee)):
                        self.update_frame(frame,pos=pose.t,rot=pose.R*0.1)
                        if i != 0 and side == "right" and hasattr(self, 'corc_response'):
                            force_data = np.array(corc_data[1+(i-1)*6:1+(i-1)*6+6])*0.01
                            self.update_line(force,points=np.vstack([pose.t, pose.t + np.matmul(pose.R,force_data[:3])]))

            self.update_response_label(self.xsens_label,f"{txt}")
        if hasattr(self, 'replayer_response'):
            print_text = self.replayer_response["print_text"]
            fps = self.replayer_response["replayer_fps"]
            self.update_response_label(self.replayer_label,f"{print_text}FPS:{fps}")
    
    """
    Cleanup
    """
    def close_worker_thread(self,worker):
        QMetaObject.invokeMethod(worker, "stop", Qt.ConnectionType.QueuedConnection)
    def close_workers(self):
        if hasattr(self, 'replayer_response'):
            self.close_worker_thread(self.replayer_worker)
    @Slot()
    def close_app(self):
        self.num_closed_threads += 1
        if self.num_closed_threads == self.num_opened_threads:
            print("Shutting down app...")
            # clear all plots
            for plt in self.plt_items:
                plt.clear()
                plt.deleteLater()
            self.close()
            self.deleteLater()

if __name__ == "__main__":
    try:
        argv = sys.argv[1]
    except:
        argv ={
               "init_flags":{"corc" :{"on":True},
                            "xsens" :{"on":True},
                            "gui"   :{
                                    "on":True,
                                    "freq":60,
                                    "3d":True,
                                    "force":True},
                            "replay":{"on":True}
                             },
               "take_num":1}
        
        argv = json.dumps(argv)

    init_args = json.loads(argv)
    app = QtWidgets.QApplication(sys.argv)
    w = ubo_replay_gui(init_args)
    w.setWindowTitle("UBO-CORC")
    w.show()

    import psutil
    # p = psutil.Process(os.getpid())
    # p.nice(psutil.REALTIME_PRIORITY_CLASS)  # or REALTIME_PRIORITY_CLASS
    sys.exit(app.exec())


    # 169.254.45.31