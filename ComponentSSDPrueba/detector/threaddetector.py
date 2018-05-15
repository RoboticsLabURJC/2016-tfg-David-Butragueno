import threading
import time
from datetime import datetime



class ThreadDetector(threading.Thread):

    def __init__(self, detector):

        self.detector = detector
        threading.Thread.__init__(self)

        self.t_cycle = 1500
        self.is_activated = False
        self.runOnce_activated = False
        self.framerate = 0

    def run(self):

        while(True):

            start_time = datetime.now()
            self.detector.update()
            end_time = datetime.now()

            dt = end_time - start_time
            dtms = ((dt.days * 24 * 60 * 60 + dt.seconds) * 1000
                + dt.microseconds / 1000.0)

            if self.is_activated:
                delta = max(self.t_cycle, dtms)
                self.framerate = float(1000.0 / delta)
            else:
                self.framerate = 9

            if(dtms < self.t_cycle):
                time.sleep((self.t_cycle - dtms) / 1000.0);

    def handleButtonDetection(self):
        self.is_activated = not self.is_activated
