import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from pca_basic_video import *
from pca_basic_utils import *
print("hi")
class runWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('실행')
        self.setGeometry(300, 300, 660, 570)
        self.initUI()
        self.video = video(self)
    
    def initUI(self):
        
        self.frame = QLabel(self)
        self.frame.resize(640,480)
        # self.frame.setScaledContents(True)
        self.frame.move(5,5)

        self.btn1 = QPushButton('켜기')
        self.btn1.setCheckable(True)
        self.btn1.clicked.connect(self.startEnd)
        self.btn2 = QPushButton('돌아가기')
        self.btn2.clicked.connect(self.returnButton_event)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.frame)
        self.vbox.addWidget(self.btn1)
        self.vbox.addWidget(self.btn2)

        self.setLayout(self.vbox)
        self.show()
        
    def startEnd(self,e):
        if self.btn1.isChecked():
            self.btn1.setText('끄기')
            try:
                self.video.startCam()
            except Exception as e:
                print("ex = ",ex)
            
        else:
            self.btn1.setText('켜기')
            self.video.stopCam()    
            

    def recvImage(self, img):        
        self.frame.setPixmap(QPixmap.fromImage(img))
        

    def returnButton_event(self):
        self.video.stopCam()
        self.close()
        self.initWindow = initWindow()

class registerWindow(QWidget):
    def __init__(self, name):
        super().__init__()
        self.setWindowTitle('Face Register')
        self.setGeometry(300, 300, 660, 570)
        self.initUI()
        self.name = name
        self.video = video(self)
    
    def initUI(self):
        
        self.frame = QLabel(self)
        self.frame.resize(640,480)
        self.frame.move(5,5)

        self.btn1 = QPushButton('학습하기')
        self.btn1.setCheckable(True)
        self.btn1.clicked.connect(self.startEnd)
        self.btn2 = QPushButton('돌아가기')
        self.btn2.clicked.connect(self.returnButton_event)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.frame)
        self.vbox.addWidget(self.btn1)
        self.vbox.addWidget(self.btn2)

        self.setLayout(self.vbox)
        self.show()
        
    def startEnd(self):
        if self.btn1.isChecked():
            self.btn1.setText('중단하기')
            try:
                self.video.startRegister(self.name)
                
                # self.close()
                # self.pro = progressWindow()  
            except Exception as e:
                print("ex.. = ",e)
            
        else:
            self.btn1.setText('학습하기')
            self.video.stopCam()    
            
            

    def recvImage(self, img):
        self.frame.setPixmap(QPixmap.fromImage(img))
            
        

    def returnButton_event(self):
        self.video.stopCam()
        self.close()
        self.initWindow = initWindow()

    def progress(self):
        self.MessageDialog()
        self.close()
        self.pro = progressWindow()

    def MessageDialog(self):
        reply = QMessageBox.question(self, 'Message', '얼굴 탐색이 완료되었습니다.\n학습이 시작됩니다.',
                                     QMessageBox.Yes)        

class progressWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Face Register')
        self.setGeometry(300, 300, 300, 200)
        self.initUI()
        self.pca = PCA()


    def initUI(self):
        
        self.pbar = QProgressBar()
        # self.pbar.setGeometry(30, 40, 200, 25)

        self.label1 = QLabel('학습 대기중 입니다.',self)
        self.label1.setAlignment(Qt.AlignCenter )  
        self.font1 = self.label1.font()
        self.font1.setPointSize(50)  

        self.btn1 = QPushButton('학습하기')
        self.btn1.setCheckable(True)
        self.btn1.clicked.connect(self.startEnd)
        self.btn2 = QPushButton('돌아가기')
        self.btn2.clicked.connect(self.returnButton_event)

        self.vbox = QVBoxLayout()
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.label1)
        self.vbox.addWidget(self.pbar)
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.btn1)
        self.vbox.addWidget(self.btn2)

        self.setLayout(self.vbox)

        self.show()

    def startEnd(self,e):
        
        if self.btn1.isChecked():
            self.pbar.setMinimum(0)
            self.pbar.setMaximum(0)
            self.label1.setText('학습 중입니다.')
            self.btn1.setText('중단하기')
            self.pca.face_register()
            self.MessageDialog()
            self.close()
            self.init = initWindow()
            
        else:
            self.btn1.setText('학습하기') 

    def returnButton_event(self,int):

        # self.video.stopCam()
        self.close()
        self.initWindow = initWindow()

    def MessageDialog(self):
        reply = QMessageBox.question(self, 'Message', '학습이 완료되었습니다.',
                                     QMessageBox.Yes)  

class initWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        registerButton = QPushButton('Face Register')
        registerButton.clicked.connect(self.registerDialog)

        recognitionButton = QPushButton('Face Recognition')
        recognitionButton.clicked.connect(self.recognitionButton_event)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(registerButton)
        vbox.addWidget(recognitionButton)
    
        vbox.addStretch(1)

        self.setLayout(vbox)

        self.setWindowTitle('PCA 얼굴인식')
        self.setGeometry(300, 300, 300, 150)
        self.show()
        

    def recognitionButton_event(self):
        self.close()
        self.runWindow = runWindow()

    
    def registerDialog(self):
    
        text, ok = QInputDialog.getText(self, '이름입력', '이름을 입력하세요')

        if ok:
            text = text.replace(" ", "")
            if(len(text) == 0):
                print("입력값 없음")
                self.nameMessageDialog()
            else:
                # print("initWindow.registerDialog name = ",text)
                self.close()
                self.registerWindow = registerWindow(text)
            
    def nameMessageDialog(self):
        reply = QMessageBox.question(self, 'Message', '이름을 다시 입력하세요',
                                     QMessageBox.Yes)  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = initWindow()
    sys.exit(app.exec_())