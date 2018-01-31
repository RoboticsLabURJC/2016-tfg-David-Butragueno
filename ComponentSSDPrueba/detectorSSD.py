import sys
from PyQt4 import QtGui
from gui.gui import Gui
from gui.threadgui import ThreadGui
from camera.camera import Camera
from camera.threadcamera import ThreadCamera
from detector.detector import Detector
from detector.threaddetector import ThreadDetector
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':

    camera = Camera()
    print(camera)

    app = QtGui.QApplication(sys.argv)
    window = Gui(camera)

    detector = Detector(camera, window)
    print(detector)

    window.setCamera(camera)
    window.show()
    window.setDetector(detector)
    window.show()

    t1 = ThreadCamera(camera)
    t1.start()

    t2 = ThreadDetector(detector)
    t2.start()

    t3 = ThreadGui(window)
    t3.start()


    sys.exit(app.exec_())
