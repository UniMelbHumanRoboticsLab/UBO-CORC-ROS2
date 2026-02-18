import os,sys
import numpy as np
import pandas as pd

from PySide6.QtCore import QObject, Signal,QTimer,Slot,QElapsedTimer,Qt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pycorc_io.package_utils.unpack_json import get_subject_params
from pycorc_io.xsens.ub_pckg.ub import ub

NUM_RFT = 3
rft_key = ["clav","ua","fa"]

class ubo_replayer(QObject):
    time_ready = Signal(dict)
    finish_save = Signal()
    stopped = Signal()
    def __init__(self,init_args,session_data):
        super().__init__()
        self.init_args = init_args
        self.take_num = session_data["take_num"]
        self.task_id = session_data["task_id"]
        self.subject_id = session_data["subject_id"]
        self.subject_path = os.path.join(os.path.dirname(__file__), '../../..',f"logs/pycorc_recordings/{self.subject_id}/{self.task_id}")
        self.body_params_rbt,self.ft_grav,self.ft_install = get_subject_params(os.path.join(os.path.dirname(__file__), '../../..',f'logs/pycorc_recordings/{self.subject_id}'))
        self.skeleton = ub(self.body_params_rbt,model="ubo",arm_side="right")
        
        # FPS Calculator
        self.replayer_frame_count = 0
        self.replayer_fps = 0
        self.replayer_timer = QElapsedTimer() # use the system clock
        self.replayer_timer.start()
        self.replayer_cur_time = self.replayer_timer.elapsed()
        self.replayer_last_time = self.replayer_timer.elapsed()
        self.elapsed_time = 0
        self.total_frames = 1000

        print("UBO Replayer Started")
        self.read_logged_data()
    def read_logged_data(self):
        # read logged data
        self.data = pd.read_csv(f"{self.subject_path}/raw/UBORecord{self.take_num}Log.csv")
        self.total_frames = len(self.data) if self.init_args["corc"]["on"]==1 else 1000
        self.frame_id = 0
        if self.init_args["corc"]["on"]:
            self.corc_data = self.data[["elapsed_time","F1x", "F1y", "F1z", "T1x", "T1y", "T1z",
                    "F2x", "F2y", "F2z", "T2x", "T2y", "T2z",
                    "F3x", "F3y", "F3z", "T3x", "T3y", "T3z"]].values
            
        if self.init_args["xsens"]["on"]:
            joints = ['trunk_ie','trunk_aa','trunk_fe',
                        'clav_dep_ev','clav_prot_ret',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps',
                        'wrist_fe','wrist_dev']
            self.right_xsens_data = self.data[[f"{joint}_right" for joint in joints]].values  # last 7 columns are CORC data
            self.left_xsens_data = self.data[[f"{joint}_left" for joint in joints]].values  # last 7 columns are CORC data
            self.timecode = self.data[["timecode"]].values
    """
    Replayer Callback
    """
    def return_data(self,print_text):
        """
        process corc data to remove gravity and bias
        """
        corc_data = self.corc_data[self.frame_id,:]
        right = self.right_xsens_data[self.frame_id,:]
        left = self.left_xsens_data[self.frame_id,:]
        robot_ee = self.skeleton.fkine(right)

        processed = [0]
        for i in range(NUM_RFT):
            # get initial_bias
            offset2 = np.array(self.ft_install[rft_key[i]])
            force_data = np.array(corc_data[1+i*6:1+i*6+6])+np.array(offset2)
            
            # weight compensate with if robot_ee exist
            pose = robot_ee[i + 1]
            weight_comp = [x * self.ft_grav[rft_key[i]] for x in [0,0,-1]]
            force_data = force_data - np.array(list(np.matmul(pose.R.T, np.array(weight_comp))) + [0,0,0])
            processed += force_data.tolist()
        processed = np.array(processed)
        
        data = {
            "print_text":print_text,
            "replayer_fps":0,
            "frame_id": self.frame_id,
            "xsens": {
                "timecode":self.timecode[self.frame_id,:],
                "right": right,
                "left": left,
            } if self.init_args["xsens"]["on"] else None ,
            "corc": {"raw_data":processed} if self.init_args["corc"]["on"] else None,
        }
        return data
    def replay_current_data(self):
        self.frame_id += 1
        if self.frame_id >= self.total_frames:
            self.frame_id = 0
            self.replayer_start_time = self.replayer_timer.elapsed()

        # update FPS
        print_text = f"Replaying {self.take_num}\n"
        self.replayer_frame_count += 1
        self.replayer_cur_time = self.replayer_timer.elapsed()
        self.elapsed_time = (self.replayer_cur_time-self.replayer_start_time)/1000
        print_text += f"Logger Time Elapsed:{self.elapsed_time}\n"

        if self.replayer_cur_time-self.replayer_last_time >= 1000:
            self.replayer_fps = self.replayer_frame_count * 1000 / (self.replayer_cur_time-self.replayer_last_time)
            self.replayer_last_time = self.replayer_cur_time
            self.replayer_frame_count = 0

        # emit the signal
        data = self.return_data(print_text)
        self.time_ready.emit(data)
    
    """
    Initialization Callback
    """
    @Slot()
    def start_take(self):
        
        
        self.poll_timer = QTimer()
        self.poll_timer.setTimerType(Qt.PreciseTimer)
        self.poll_timer.timeout.connect(self.replay_current_data)
        self.poll_timer.start(int(1/100*1000))
        self.replayer_start_time = self.replayer_timer.elapsed()

    """
    External Signals Callbacks
    """
    @Slot()
    def stop(self):
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        self.stopped.emit()
    @Slot()
    def stop_take(self):
        if hasattr(self, 'poll_timer'):
            # emit the signal
            data = self.return_data(f"Stop Take {self.take_num}\n")
            self.time_ready.emit(data)
            self.poll_timer.stop()
    @Slot()
    def next_take(self):
        if hasattr(self, 'poll_timer'):
            try:
                self.take_num += 1
                self.read_logged_data()
            except Exception as e:
                self.take_num = 1
                self.read_logged_data()
                print(f"Restart {self.task_id}")
            finally:
                self.replayer_start_time = self.replayer_timer.elapsed()



        