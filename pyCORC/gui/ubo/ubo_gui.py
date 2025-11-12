import os, sys,json

from ubo_logger import ubo_logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.pycorc_gui import pycorc_gui

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
print(sys.path)
from pycorc_io.corc.corc_FLNL_client import corc_FLNL_client

import numpy as np
from scipy.spatial.transform import Rotation as R
from PySide6 import QtWidgets
from PySide6.QtGui import QShortcut
from PySide6.QtCore import QThread,Qt,QMetaObject,Slot

NUM_RFT = 1
class ubo_gui(pycorc_gui):
    def __init__(self,init_args):
        self.init_args = init_args
        self.corc_args = self.init_args["init_flags"]["corc"]
        self.gui_args = self.init_args["init_flags"]["gui"]
        self.log_args = self.init_args["init_flags"]["log"]

        self.num_closed_threads = 0
        self.num_opened_threads = 0

        super().__init__(freq=self.gui_args["freq"],gui_3d=self.gui_args["3d"])

        if self.gui_args["force"]:
            self.ubo_wrenches_live_stream = self.init_live_stream(num_columns=2)
            
        # import debugpy
        # debugpy.listen(("localhost", 5678))
        # print("Waiting for debugger…")
        # debugpy.wait_for_client()
        """
        Initilization
        """
        self.init_IOs()
        self.init_shortcuts()

    """
    Init Sensor Worker / Thread / GUI Helper Functions
    """
    def add_opened_threads(self):
        self.num_opened_threads += 1
        print(f"Threads Opened:{self.num_opened_threads}")
    def init_corc(self):
        # init response label
        self.corc_label = self.init_response_label(size=[500,50])
        # init corc thread and worker
        self.corc_thread = QThread()
        self.corc_worker = corc_FLNL_client(ip=self.corc_args["ip"],port=self.corc_args["port"])
        # init corc live stream plot
        self.ubo_F_live_stream_plots = []
        self.ubo_M_live_stream_plots = []
        if self.gui_args["on"] and self.gui_args["force"]:
            for i in range(NUM_RFT):
                self.ubo_F_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.ubo_wrenches_live_stream,sensor_name= f"UBO_{i} Force",unit="N",dim=3))
                self.ubo_M_live_stream_plots.append(self.add_live_stream_plot(live_stream=self.ubo_wrenches_live_stream,sensor_name= f"UBO_{i} Moment",unit="Nm",dim=3))
    def init_logger(self):
        # init response label
        self.logger_label = self.init_response_label(size=[500,150])
        self.logger_thread = QThread()
        self.logger_worker = ubo_logger(sensor_flags=self.init_args["init_flags"],
                                            side=self.init_args["side"],
                                            finger_id=[self.init_args["finger_name"],self.init_args["finger_id"]],
                                            take_num = self.init_args["take_num"] )
    def init_shortcuts(self):
        # to start / stop logging at button press
        if self.log_args["on"]:
            # to start logging
            start_log = QShortcut("S", self)
            start_log.activated.connect(self.logger_worker.start_worker)
            # to stop logging and close everything
            stop_log = QShortcut("C", self)
            stop_log.activated.connect(self.logger_worker.reset_logger)
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
        if self.log_args["on"]:
            self.init_logger()
            self.logger_response = {
            "print_text":"Idle\n",
            "logger_fps":0.0
        }
        
        """
        Move sensor worker to thread 
        Connect intersensor signals
        Connect signals to sensor GUI callbacks
        Connect thread start to start sensor worker, update number of open threads
        Connect worker stop to stop thread at closeup
        Connect thread finished to close app
        """
        if self.corc_args["on"]:
            # move worker to thread
            self.corc_worker.moveToThread(self.corc_thread)
            # connect to sensor gui
            self.corc_worker.data_ready.connect(self.update_corc,type=Qt.ConnectionType.QueuedConnection)
            # connect thread start to start worker   
            self.corc_thread.started.connect(self.corc_worker.start_worker)
            self.corc_thread.started.connect(self.add_opened_threads)
            # connect worker stop to stop thread at closeup
            self.corc_worker.stopped.connect(self.corc_thread.exit)
            # connect thread finished to close app
            self.corc_thread.finished.connect(self.close_app)
        if self.log_args["on"]:
            # move worker to thread
            self.logger_worker.moveToThread(self.logger_thread)
            # connect rft to logger
            if self.init_args["init_flags"]["rft"]  == 1:
                self.corc_worker.forces_ready.connect(self.logger_worker.update_rft,type=Qt.ConnectionType.QueuedConnection)
            # connect logger to sensor gui
            self.logger_worker.time_ready.connect(self.update_logger,type=Qt.ConnectionType.QueuedConnection)
            # connect thread start to add opened threads  
            self.logger_thread.started.connect(self.add_opened_threads)
            # connect logger finished saving to close workers
            self.logger_worker.finish_save.connect(self.close_workers)
            # connect worker stop to stop thread at closeup
            self.logger_worker.stopped.connect(self.logger_thread.exit)
            # connect thread finished to close app
            self.logger_thread.finished.connect(self.close_app)

        """
        Start threads
        """
        if self.corc_args["on"]:
            self.corc_thread.start()
            self.corc_thread.setPriority(QThread.Priority.TimeCriticalPriority)
        if self.log_args["on"]:
            self.logger_thread.start()
            self.logger_thread.setPriority(QThread.Priority.TimeCriticalPriority)
        
    """
    Update Sensor Variable Callbacks
    """
    @Slot(dict)
    def update_corc(self,corc_response):
        self.corc_response = corc_response
    @Slot(dict)
    def update_logger(self,logger_response):
        self.logger_response = logger_response

    """
    Update Main GUI Helper Functions and Callback
    """
    def update_gui(self):
        super().update_gui()
        if self.gui_args["on"]:
            # update live stream
            if self.gui_args["force"]:
                self.update_live_stream_buffer(live_stream=self.ubo_wrenches_live_stream)
        if hasattr(self, 'corc_response'):
            # update rft info
            corc_data = self.corc_response["raw_data"]
            fps = self.corc_response["corc_fps"]
            self.update_response_label(self.corc_label,f"FPS:{fps}\nCORC Running Time:{corc_data[0]}s\n{corc_data[1:]}")
            if self.gui_args["on"] and self.gui_args["force"]:
                for i in range(NUM_RFT):
                    force_data = corc_data[1+i*6:1+i*6+6]
                    # pass
                    self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_F_live_stream_plots[i],force_data,dim=3)
                    self.update_live_stream_plot(self.ubo_wrenches_live_stream,self.ubo_M_live_stream_plots[i],force_data[3:],dim=3)
        if hasattr(self, 'logger_response'):
            print_text = self.logger_response["print_text"]
            fps = self.logger_response["logger_fps"]
            self.update_response_label(self.logger_label,f"{print_text}FPS:{fps}")
    
    """
    Cleanup
    """
    def close_worker_thread(self,worker):
        QMetaObject.invokeMethod(worker, "stop", Qt.ConnectionType.QueuedConnection)
    def close_workers(self):
        if hasattr(self, 'corc_response'):
            self.close_worker_thread(self.corc_worker)
        if hasattr(self, 'logger_response'):
            self.close_worker_thread(self.logger_worker)
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
               "init_flags":{"corc":{"on":True,
                                     "ip":"127.0.0.1",
                                     "port":2048},
                             "gui":{"on":True,
                                    "freq":60,
                                    "3d":False,
                                    "force":True},
                             "log":{"on":False}
                             },
               "take_num":0}
        
        argv = json.dumps(argv)

    init_args = json.loads(argv)
    app = QtWidgets.QApplication(sys.argv)
    w = ubo_gui(init_args)
    w.setWindowTitle("UBO-CORC")
    w.show()

    import psutil
    # p = psutil.Process(os.getpid())
    # p.nice(psutil.REALTIME_PRIORITY_CLASS)  # or REALTIME_PRIORITY_CLASS
    sys.exit(app.exec())