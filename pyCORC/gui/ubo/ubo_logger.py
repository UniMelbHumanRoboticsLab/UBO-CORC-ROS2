import os
import numpy as np

import pandas as pd

from PySide6.QtCore import QObject, Signal,QTimer,Slot,QElapsedTimer,Qt
import debugpy

class ubo_logger(QObject):
    time_ready = Signal(dict)
    finish_save = Signal()
    stopped = Signal()
    def __init__(self,init_args,session_data):
        super().__init__()
        self.init_args = init_args
        self.take_num = session_data["take_num"]
        self.task_id = session_data["task_id"]
        self.subject_id = session_data["subject_id"]
        self.save_path = os.path.join(os.path.dirname(__file__), '../../..',f"logs/pycorc_recordings/{self.subject_id}/{self.task_id}")

        # FPS Calculator
        self.logger_frame_count = 0
        self.logger_fps = 0
        self.logger_timer = QElapsedTimer() # use the system clock
        self.logger_timer.start()
        self.logger_cur_time = self.logger_timer.elapsed()
        self.logger_last_time = self.logger_timer.elapsed()
        self.elapsed_time = 0
        
        print("UBO Logger Started")
    
    """
    Logging Callback
    """
    def log_current_data(self):
        # update FPS
        print_text = f"Logging Take {self.take_num+1}\n"
        self.logger_frame_count += 1
        self.logger_cur_time = self.logger_timer.elapsed()
        self.elapsed_time = (self.logger_cur_time-self.logger_start_time)/1000
        print_text += f"Time Elapsed:{self.elapsed_time}\n"
        if self.logger_cur_time-self.logger_last_time >= 1000:
            self.logger_fps = self.logger_frame_count * 1000 / (self.logger_cur_time-self.logger_last_time)
            self.logger_last_time = self.logger_cur_time
            self.logger_frame_count = 0

        if self.init_args["corc"]["on"] and hasattr(self, 'corc_response'):
            print_text += f'CORC:{self.corc_response["corc_fps"]}\n'
            self.corc_arr.append(self.corc_response["raw_data"]+[self.elapsed_time])

        if self.init_args["xsens"]["on"] and hasattr(self, 'xsens_response'):
            print_text += f'XSENS:{self.xsens_response["xsens_fps"]}\n'

            timecode = self.xsens_response["timecode"]
            right = self.xsens_response["right"]["list"]
            left = self.xsens_response["left"]["list"]
            # print(timecode)
            self.xsens_arr.append([timecode]+right+left)

        # emit the signal
        data = {
            "print_text":print_text,
            "logger_fps":self.logger_fps
        }
        self.time_ready.emit(data)
    
    """
    Initialization Callback
    """
    @Slot()
    def start_worker(self):
        self.poll_timer = QTimer()
        self.poll_timer.setTimerType(Qt.PreciseTimer)
        self.poll_timer.timeout.connect(self.log_current_data)
        self.poll_timer.start(int(1/100*1000))
        self.logger_start_time = self.logger_timer.elapsed()

        # corc
        if self.init_args["corc"]["on"]:
            self.corc_arr = []
        # xsens
        if self.init_args["xsens"]["on"]:
            self.xsens_arr = []
        
    """
    External Signals Callbacks
    """
    @Slot(dict)
    def update_corc(self,corc_response): # [self.rft_data_arr,self.rft_pose,self.rft_fps]
        self.corc_response = corc_response
    @Slot(dict)
    def update_xsens(self,xsens_response): # [self.rft_data_arr,self.rft_pose,self.rft_fps]
        self.xsens_response = xsens_response
    @Slot()
    def stop(self):
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        self.stopped.emit()
    @Slot()
    def reset_logger(self):
        # debugpy.debug_this_thread()
        if hasattr(self, 'poll_timer'):
            self.poll_timer.stop()
        
        print(f"Elapsed Time: {self.elapsed_time}")
        # emit the signal
        self.time_ready.emit({
            "print_text":"Logger Saving \n",
            "logger_fps": 0.0
        })

        if self.init_args["corc"]["on"]:
            if hasattr(self, 'corc_response'):
                # joints = ['trunk_ie','trunk_aa','trunk_fe',
                #           'scapula_de','scapula_pr',
                #           'shoulder_fe','shoulder_aa','shoulder_ie',
                #           'elbow_fe','elbow_ps',
                #           'wrist_fe','wrist_dev']
                # xsens_column = ["timecode"]+[f"{joint}_{side}" for side in ["right","left"] for joint in joints]
                corc_column = [
                            "corc time",
                            "F1x","F1y","F1z","T1x","T1y","T1z",
                            "F2x","F2y","F2z","T2x","T2y","T2z",
                            "F3x","F3y","F3z","T3x","T3y","T3z",
                            "elapsed_time"
                           ]
                # total_arr = np.hstack((np.array(self.xsens_arr),np.array(self.corc_arr)))
                # df = pd.DataFrame(total_arr,columns=xsens_column+corc_column)
                
                total_arr = np.array(self.corc_arr)
                df = pd.DataFrame(total_arr,columns=corc_column)
                
                self.save_file(path=f"{self.save_path}/",df=df,item=f"UBORecord{self.take_num+1}Log")
        # emit the signal
        self.time_ready.emit({
            "print_text":"Logger Stopped \n",
            "logger_fps": 0.0
        })
        self.take_num += 1
        
    """
    Helper Function
    """    
    def save_file(self,path,df:pd.DataFrame,item):
        if not os.path.isdir(path):
            os.makedirs(path)
        df.to_csv(f"{path}/{item}.csv",index=False)    

    