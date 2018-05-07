import threading
import time
from datetime import datetime

t_cycle = 1500 # ms

class ThreadDetector(threading.Thread):

    def __init__(self, detector):

        self.detector = detector
        threading.Thread.__init__(self)

        self.is_activated = False

    def run(self):

        while(True):

            start_time = datetime.now()
            self.detector.update()
            end_time = datetime.now()

            dt = end_time - start_time
            dtms = ((dt.days * 24 * 60 * 60 + dt.seconds) * 1000
                + dt.microseconds / 1000.0)

            if(dtms < t_cycle):
                time.sleep((t_cycle - dtms) / 1000.0);

    def handleButtonDetection(self):
        self.is_activated = not self.is_activated
