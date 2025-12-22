import os
import numpy as np

import pandas as pd

from PySide6.QtCore import QObject, Signal,QTimer,Slot,QElapsedTimer,Qt
import debugpy

class ubo_replayer(QObject):
    time_ready = Signal(dict)
    finish_save = Signal()
    stopped = Signal()
    def __init__(self,init_args,take_num):
        super().__init__()
        self.init_args = init_args
        self.take_num = take_num
        self.save_path = "./logs/pycorc_recordings/"

        # FPS Calculator
        self.replayer_frame_count = 0
        self.replayer_fps = 0
        self.replayer_timer = QElapsedTimer() # use the system clock
        self.replayer_timer.start()
        self.replayer_cur_time = self.replayer_timer.elapsed()
        self.replayer_last_time = self.replayer_timer.elapsed()
        self.elapsed_time = 0

        # read logged data
        self.data = pd.read_csv(f"{self.save_path}/exp1/p1/vincent/task_1/var_2/UBORecord{self.take_num}Log.csv")
        if self.init_args["corc"]["on"]:
            self.corc_data = self.data[["corc time","F1x", "F1y", "F1z", "T1x", "T1y", "T1z",
                    "F2x", "F2y", "F2z", "T2x", "T2y", "T2z",
                    "F3x", "F3y", "F3z", "T3x", "T3y", "T3z"]].values
            
        if self.init_args["xsens"]["on"]:
            joints = ['trunk_ie','trunk_aa','trunk_fe',
                        'scapula_de','scapula_pr',
                        'shoulder_fe','shoulder_aa','shoulder_ie',
                        'elbow_fe','elbow_ps',
                        'wrist_fe','wrist_dev']
            self.right_xsens_data = self.data[[f"{joint}_right" for joint in joints]].values  # last 7 columns are CORC data
            self.left_xsens_data = self.data[[f"{joint}_left" for joint in joints]].values  # last 7 columns are CORC data
        
        self.frame_id = 0
        self.total_frames = len(self.data) if self.init_args["corc"]["on"]==1 else 1000
        print("UBO Replayer Started")
    
    """
    Logging Callback
    """
    def replay_current_data(self):
        self.frame_id += 1
        if self.frame_id >= self.total_frames:
            self.frame_id = 0
            self.replayer_start_time = self.replayer_timer.elapsed()

        # update FPS
        print_text = f"Logging\n"
        self.replayer_frame_count += 1
        self.replayer_cur_time = self.replayer_timer.elapsed()
        self.elapsed_time = (self.replayer_cur_time-self.replayer_start_time)/1000
        print_text += f"Time Elapsed:{self.elapsed_time}\n"


        if self.replayer_cur_time-self.replayer_last_time >= 1000:
            self.replayer_fps = self.replayer_frame_count * 1000 / (self.replayer_cur_time-self.replayer_last_time)
            self.replayer_last_time = self.replayer_cur_time
            self.replayer_frame_count = 0

        # emit the signal
        data = {
            "print_text":print_text,
            "replayer_fps":self.replayer_fps,
            "frame_id": self.frame_id,
            "xsens": {
                "right": self.right_xsens_data[self.frame_id,:],
                "left": self.left_xsens_data[self.frame_id,:],
            } if self.init_args["xsens"]["on"] else None ,
            "corc": {"raw_data":self.corc_data[self.frame_id,:]} if self.init_args["corc"]["on"] else None,
        }
        self.time_ready.emit(data)
    
    """
    Initialization Callback
    """
    @Slot()
    def start_worker(self):
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
    def stop_replayer(self):
        # debugpy.debug_this_thread()
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        
        # emit the signal
        data = {
            "print_text":"Finish Replay",
            "replayer_fps":self.replayer_fps,
            "frame_id": self.frame_id,
            "xsens": {
                "right": None,
                "left": None
            },
            "corc": {"raw_data":self.corc_data[self.frame_id,:]} if self.init_args["corc"]["on"] else None,
        }
        self.finish_save.emit(data)
        