import os
import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)
import pandas as pd

from PySide6.QtCore import QObject, Signal,QTimer,Slot,QElapsedTimer,Qt
import debugpy

class ubo_logger(QObject):
    time_ready = Signal(dict)
    finish_save = Signal()
    stopped = Signal()
    def __init__(self,sensor_flags,side,finger_id,take_num):
        super().__init__()
        self.sensor_flags = sensor_flags
        self.take_num = take_num
        self.finger_name = finger_id[0]
        self.finger_id = finger_id[1]
        self.sensor_path = f"./sensor_calib_fsr/data/{side}/{self.finger_name}/"

        # FPS Calculator
        self.logger_frame_count = 0
        self.logger_fps = 0
        self.logger_timer = QElapsedTimer() # use the system clock
        self.logger_timer.start()
        self.logger_cur_time = self.logger_timer.elapsed()
        self.logger_last_time = self.logger_timer.elapsed()
        self.elapsed_time = 0
        

        print("FMC Calib Logger Started")
    
    """
    Logging Callback
    """
    def log_current_data(self):
        
        # update FPS
        print_text = f"Logging\n"
        self.logger_frame_count += 1
        self.logger_cur_time = self.logger_timer.elapsed()
        self.elapsed_time = (self.logger_cur_time-self.logger_start_time)/1000
        print_text += f"Time Elapsed:{self.elapsed_time}\n"
        if self.logger_cur_time-self.logger_last_time >= 1000:
            self.logger_fps = self.logger_frame_count * 1000 / (self.logger_cur_time-self.logger_last_time)
            self.logger_last_time = self.logger_cur_time
            self.logger_frame_count = 0

        # log sensor
        if self.sensor_flags["esp"] == 1:
            print_text += f'ESP:{self.esp_response["esp_fps"]}\n'
            raw_data = self.esp_response["raw_data"][self.finger_id]
            self.esp_arr.append([raw_data])
        if self.sensor_flags["rft"] == 1 and hasattr(self, 'rft_response'):
            print_text += f'RFT:{self.rft_response["rft_fps"]}\n'
            mag = np.linalg.norm(self.rft_response["rft_data_arr"][:3])
            self.rft_arr.append(np.concatenate((np.array([self.elapsed_time]),self.rft_response["rft_data_arr"],np.array([mag]))))

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

        # hand
        if self.sensor_flags["esp"] == 1:
            self.esp_arr = []
        # rft
        if self.sensor_flags["rft"] == 1:
            self.rft_arr = []
        
    """
    External Signals Callbacks
    """
    @Slot(dict)
    def update_esp(self,esp_response):
        self.esp_response = esp_response
    @Slot(dict)
    def update_rft(self,rft_response): # [self.rft_data_arr,self.rft_pose,self.rft_fps]
        self.rft_response = rft_response
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

        if self.sensor_flags["esp"] == 1 and self.sensor_flags["rft"] == 1:
            if hasattr(self, 'esp_response') and hasattr(self, 'rft_response'):
                columns = ["time","Fx","Fy","Fz","Tx","Ty","Tz","F","1/R"]
                total_arr = np.hstack((np.array(self.rft_arr),np.array(self.esp_arr)))
                df = pd.DataFrame(total_arr,columns=columns)
                self.save_file(path=f"{self.sensor_path}/",df=df,item=f"force_{self.take_num}")
        # emit the signal
        self.time_ready.emit({
            "print_text":"Logger Stopped \n",
            "logger_fps": 0.0
        })
        self.finish_save.emit()
        
    """
    Helper Function
    """    
    def save_file(self,path,df:pd.DataFrame,item):
        if not os.path.isdir(path):
            os.makedirs(path)
        df.to_csv(f"{path}/{item}.csv",index=False)    

    