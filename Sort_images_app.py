import os
import time
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from PIL import Image, ImageQt
from threading import Thread
from object_detection_tf.utils.extra_tools import SSD_Model_Names,\
        RCNN_Model_Names

from utils import detect_visualize, get_files_path, extentions


class ComboBox(QtWidgets.QComboBox):
    clicked = QtCore.pyqtSignal()
    def showPopup(self):
        self.clicked.emit()
        return super(ComboBox, self).showPopup()
QtWidgets.QComboBox = ComboBox


class Ui_MainWindow(QtCore.QObject):

    progressSignal1 = QtCore.pyqtSignal(int)
    progressSignal2 = QtCore.pyqtSignal(int)
    imageSignal = QtCore.pyqtSignal(np.ndarray)

    threshold_value = 0.7
    est_avg_time = []
    model_name = 'SSDLITE Mobilenet V2'

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Image Sorting")
        MainWindow.setWindowTitle("Image Sorting")
        MainWindow.resize(618, 669)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(240, 0, 140, 41))
        self.label_3.setStyleSheet("")
        self.label_3.setObjectName("label_3")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(240, 630, 140, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setDisabled(1)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(280, 360, 61, 41))
        self.label_4.setStyleSheet("")
        self.label_4.setObjectName("label_4")

        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(260, 430, 90, 20))
        self.checkBox.setStyleSheet("color: rgb(255, 255, 255);")
        self.checkBox.setObjectName("checkBox")

        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 340, 601, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.dial = QtWidgets.QDial(self.centralwidget)
        self.dial.setGeometry(QtCore.QRect(40, 390, 111, 111))
        self.dial.setObjectName("dial")
        self.dial.setValue( self.dial.maximum() )
        self.dial.setDisabled(1)
        self.dial_label = QtWidgets.QLabel(self.centralwidget)
        self.dial_label.setGeometry(QtCore.QRect(40, 490, 110, 20))
        self.dial_label.setStyleSheet("Color:White;")
        self.dial_label.setObjectName("dial_label")

        self.threshold_slider = QtWidgets.QSlider(self.centralwidget)
        self.threshold_slider.setGeometry(QtCore.QRect(450, 415, 130, 20))
        self.threshold_slider.setStyleSheet("color: white")
        self.threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.threshold_slider.setObjectName("threshold_slider")
        self.threshold_slider.setRange(0, 1000)
        self.threshold_slider.setValue(700)
        self.threshold_label = QtWidgets.QLabel(self.centralwidget)
        self.threshold_label.setGeometry(QtCore.QRect(450, 400, 130, 16))
        self.threshold_label.setStyleSheet("color:white")
        self.threshold_label.setObjectName("threshold_label")

        self.showvisualckb = QtWidgets.QCheckBox(self.centralwidget)
        self.showvisualckb.setGeometry(QtCore.QRect(450, 440, 140, 20))
        self.showvisualckb.setStyleSheet("color: white")
        self.showvisualckb.setObjectName("showvisualckb")

        self.savevisualckb = QtWidgets.QCheckBox(self.centralwidget)
        self.savevisualckb.setGeometry(QtCore.QRect(450, 460, 140, 20))
        self.savevisualckb.setStyleSheet("color: white")
        self.savevisualckb.setObjectName("savevisualckb")

        self.nestedfldrckb = QtWidgets.QCheckBox(self.centralwidget)
        self.nestedfldrckb.setGeometry(QtCore.QRect(450, 480, 140, 20))
        self.nestedfldrckb.setStyleSheet("color: white")
        self.nestedfldrckb.setObjectName("nestedfldrckb")

        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 540, 601, 80))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(430, 40, 141, 32))
        self.pushButton_2.setObjectName("pushButton_2")

        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(30, 10, 351, 31))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(30, 40, 351, 31))
        self.label_2.setObjectName("label_2")

        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(430, 10, 141, 32))
        self.pushButton.setObjectName("pushButton")

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(210, 400, 200, 20))
        self.lineEdit.setObjectName("lineEdit")

        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(40, 40, 540, 300))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")

        self.photo_label = QtWidgets.QLabel(self.frame_2)
        self.photo_label.setGeometry(QtCore.QRect(0, 0, 540, 300))
        self.photo_label.setStyleSheet("background-color: rgb(57, 56, 58); border: 1px solid rgb(15, 15, 15);")
        self.photo_label.setText("")
        self.photo_label.setObjectName("photo_label")
        self.photo_label.setAlignment(QtCore.Qt.AlignCenter)

        self.progressBar = QtWidgets.QProgressBar(self.frame_2)
        self.progressBar.setGeometry(QtCore.QRect(10, 290, 521, 10))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.hide()

        self.estTime_label = QtWidgets.QLabel(self.frame_2)
        self.estTime_label.setGeometry(QtCore.QRect(10, 270, 521, 30))
        self.estTime_label.setObjectName("estTime_label")
        self.estTime_label.hide()

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(150, 510, 320, 26))
        self.comboBox.setObjectName("comboBox")

        for dictionary in (SSD_Model_Names, RCNN_Model_Names):
            for k, v in dictionary.items():
                text = '✓ ( {}ms,  {}mAP )  |  {}'.format(v[1][0], v[1][1], k)
                self.comboBox.addItem(text)
        self.comboBox.setCurrentIndex(9)
        
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(260, 480, 101, 20))
        self.label_6.setStyleSheet("color:white")
        self.label_6.setObjectName("label_6")

        self.frame_2.raise_()
        self.label_3.raise_()
        self.pushButton_3.raise_()
        self.label_4.raise_()
        self.checkBox.raise_()
        self.line.raise_()
        self.dial.raise_()
        self.dial_label.raise_()
        self.threshold_slider.raise_()
        self.showvisualckb.raise_()
        self.frame.raise_()
        self.lineEdit.raise_()
        self.comboBox.raise_()
        self.label_6.raise_()
        self.threshold_label.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.connectUi()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:600; color:#ffffff;\">Image Preview</span></p></body></html>"))
        self.pushButton_3.setText(_translate("MainWindow", "Start"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600; color:#ffffff;\">Search</span></p></body></html>"))
        self.checkBox.setText(_translate("MainWindow", "All Objects"))
        self.dial_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Image: MAX</p></body></html>"))
        self.showvisualckb.setText(_translate("MainWindow", "Show Visualization"))
        self.savevisualckb.setText(_translate("MainWindow", "Save Visualization"))
        self.nestedfldrckb.setText(_translate("MainWindow", "Go through folders"))
        self.pushButton_2.setText(_translate("MainWindow", "Save Directory"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt; color:#ffffff;\">Select directory path display</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt; color:#ffffff;\">Save directory path display</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Select Directory"))
        self.estTime_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:11pt;\">est: calculating..</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600;\">Select model</span></p></body></html>"))
        self.threshold_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Threshold: 70.0%</p></body></html>"))

    def connectUi(self):
        self.imageSignal.connect( self.updateImage )
        self.comboBox.clicked.connect( self.updateComboBox )
        self.dial.valueChanged.connect( self.setImageAmount )
        self.progressSignal1.connect( self.progressBar.setValue )
        self.progressSignal2.connect( self.progressBar.setMaximum )
        self.comboBox.currentTextChanged.connect(self.updateModelName)
        self.threshold_slider.valueChanged.connect(self.updateThreshold)
        self.pushButton.clicked.connect(lambda: self.setDirectory(self.label))
        self.pushButton_2.clicked.connect(lambda: self.setDirectory(self.label_2))
        self.pushButton_3.clicked.connect(lambda: Thread(
            target=self.actualDetection, kwargs={ 'select_path': self.label.text(), 
            'save_path': self.label_2.text(), 'search':self.lineEdit.text()
            }, daemon=True ).start() )

    def updateModelName(self, name=None):
        self.model_name = name.split('  |  ')[-1]

    def updateComboBox(self):
        downloaded_models = []
        models_dict = {**SSD_Model_Names, **RCNN_Model_Names}
        for name, model in models_dict.items():
            if model[0] in os.listdir('COCO-trained_models'):
                downloaded_models.append(name)
        
        # self.comboBox.clear()
        for count, (k, v) in enumerate(models_dict.items()):
            if k in downloaded_models:
                text = '| ✓ ( {}ms,  {}mAP )  |  {}'.format(v[1][0], v[1][1], k)
            else:
                text = '| ⬇ ( {}ms,  {}mAP )  |  {}'.format(v[1][0], v[1][1], k)
            self.comboBox.setItemText(count, text)
        # print(self.comboBox.currentText().split('  |  ')[-1])

    def setImageAmount(self, value):
        self.dial_label.setText("<html><head/><body><p align=\"center\">Image: \
            {}</p></body></html>".format(value))

    def updateThreshold(self, value):
        self.threshold_label.setText('<html><head/><body><p align=\"center\">Threshold: \
            {}%</p></body></html>'.format(str(value/10)))
        self.threshold_value = value/1000

    def setDirectory(self, label: QtWidgets.QLabel):
        """Set Label directory bt label.setText."""
        _translate = QtCore.QCoreApplication.translate
        options = QtWidgets.QFileDialog.DontResolveSymlinks | QtWidgets.QFileDialog.ShowDirsOnly
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "Open Folder" ,options=options)
        if directory: 
            label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" \
                font-size:11pt; color:#ffffff;\">{}</span></p></body></html>".format(directory)))
            img_list=[]
            if self.nestedfldrckb.isChecked():
                img_list = get_files_path(directory, False, *extentions)
            else:
                for img in os.listdir(directory):
                    if not os.path.isdir(directory+'/'+img) and not img.startswith('.'):
                        img_list.append(img)
            self.dial.setMaximum( len(img_list) )
            self.dial.setValue( self.dial.maximum() )
            self.pushButton_3.setEnabled(1)
            self.dial.setEnabled(1)
    
    def updateImage(self, array):
        img = QtGui.QImage(array, array.shape[1],array.shape[0], 
            array.shape[1] * 3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(img)
        pixmap = pixmap.scaled(self.photo_label.width(), self.photo_label.height(), 
                        QtCore.Qt.KeepAspectRatio
                        )
        self.photo_label.setPixmap(pixmap)

    def actualDetection(self, select_path, save_path=None, search=None):
        self.pushButton_3.setDisabled(1)
        self.checkBox.setDisabled(1)
        self.lineEdit.setDisabled(1)
        self.frame.setDisabled(1)
        self.dial.setDisabled(1)
        self.progressBar.show()
        self.estTime_label.show()

        doc = QtGui.QTextDocument()
        doc.setHtml(select_path)
        select_path = doc.toPlainText()
        doc.setHtml(save_path)
        save_path = doc.toPlainText()

        if save_path is not None and save_path == 'Save directory path display':
            save_path = select_path
            _translate = QtCore.QCoreApplication.translate
            self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" \
                font-size:11pt; color:#ffffff;\">{}</span></p></body></html>".format(save_path)))
                
        img_list=[]
        incri_by = 0
        if self.nestedfldrckb.isChecked():
            img_list = get_files_path(select_path, False, *extentions)
        else:
            for img in os.listdir(select_path):
                if not os.path.isdir(select_path+'/'+img) and not img.startswith('.'):
                    img_list.append(os.path.join(select_path, img))

        img_list = img_list[ :self.dial.value() ]
        self.progressSignal2.emit( len(img_list) )

        for img_path in img_list:
            try:
                img = img_path.split('/')[-1]
                start_timne = time.time()
                incri_by += 1
                p_img = Image.open(img_path)
                print(np.array(p_img).shape)
                np_img, name_list = detect_visualize(np.array(p_img), 
                                        self.showvisualckb.isChecked(), 
                                        self.threshold_value, 
                                        self.model_name)
                for name, score in name_list:
                    if not self.checkBox.isChecked() and search is not None:
                        if search.lower() == name.lower():
                            self.imageSignal.emit(np_img)
                            if not os.path.exists(save_path+'/sorted/'+name):
                                os.makedirs(save_path+'/sorted/'+name)
                            if self.savevisualckb.isChecked():
                                p1_img = Image.fromarray(np_img)
                                p1_img.save(save_path+'/sorted/'+name+'/'+img)
                            else: 
                                p1_img = Image.fromarray(np.array(p_img))
                                p1_img.save(save_path+'/sorted/'+name+'/'+img)
                        else:
                            self.imageSignal.emit(np.array(p_img))
                            
                    elif self.checkBox.isChecked():
                        self.imageSignal.emit(np_img)
                        if not os.path.exists(save_path+'/sorted/'+name):
                            os.makedirs(save_path+'/sorted/'+name)
                        if self.savevisualckb.isChecked():
                            p1_img = Image.fromarray(np_img)
                            p1_img.save(save_path+'/sorted/'+name+'/'+img)
                        else: 
                            p1_img = Image.fromarray(np.array(p_img))
                            p1_img.save(save_path+'/sorted/'+name+'/'+img)
                    
                print(img, incri_by)
                self.progressSignal1.emit(incri_by)
            
                self.est_avg_time.append( time.time() - start_timne )
                if self.est_avg_time.__len__() > 20:
                    self.est_avg_time.pop(0)

                est_time = (sum(self.est_avg_time) / self.est_avg_time.__len__()) \
                                * (self.dial.value() - incri_by)
                self.estTime_label.setText("<html><head/><body><p align=\"center\"><span \
                    style=\" font-size:11pt;\">est: {:.2f} second(s)</span></p></body></html>".
                    format(est_time))

            except Exception as e: 
                print(e)
        
        self.pushButton_3.setEnabled(1)
        self.lineEdit.setEnabled(1)
        self.checkBox.setEnabled(1)
        self.frame.setEnabled(1)
        self.dial.setEnabled(1)
        self.progressBar.hide()
        self.estTime_label.hide()
        self.progressBar.setValue(0)

        self.estTime_label.setText("<html><head/><body><p align=\"center\"><span\
             style=\" font-size:11pt;\">est: calculating..</span></p></body></html>")
        self.photo_label.clear()
        return 




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
