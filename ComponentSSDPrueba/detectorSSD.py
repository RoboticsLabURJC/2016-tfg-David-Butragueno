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

    t_cam = ThreadCamera(camera)
    t_cam.start()

    t_detector = ThreadDetector(detector)
    t_detector.start()

    window.setCamera(camera, t_cam)
    window.show()
    window.setDetector(detector, t_detector)
    window.show()

    t_gui = ThreadGui(window)
    t_gui.start()


    sys.exit(app.exec_())
