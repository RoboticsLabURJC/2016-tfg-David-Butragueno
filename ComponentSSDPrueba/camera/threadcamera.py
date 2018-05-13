import threading
import time
from datetime import datetime

class ThreadCamera(threading.Thread):

    def __init__(self, camera):
        self.camera = camera
        threading.Thread.__init__(self)

        self.t_cycle = 1500 # ms

    def run(self):

        while(True):

            start_time = datetime.now()
            self.camera.update()
            end_time = datetime.now()

            dt = end_time - start_time
            dtms = ((dt.days * 24 * 60 * 60 + dt.seconds) * 1000
                + dt.microseconds / 1000.0)

            delta = max(self.t_cycle, dtms)
            self.framerate = float(1000.0 / delta)
            print self.framerate

            if(dtms < self.t_cycle):
                time.sleep((self.t_cycle - dtms) / 1000.0);
