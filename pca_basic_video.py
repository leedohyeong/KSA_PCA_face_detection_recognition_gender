import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from threading import Thread
from imutils import face_utils

from pca_basic_utils import *
import time , os, dlib
import numpy as np


class video(QObject):

    sendImage = pyqtSignal(QImage)
    prograss_run = pyqtSignal()

    # load model
    model_path = 'models/opencv_face_detector_uint8.pb'
    config_path = 'models/opencv_face_detector.pbtxt'
    predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
    net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
    predictor = dlib.shape_predictor(predictor_path)

    conf_threshold = 0.5
        
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.sendImage.connect(self.widget.recvImage)
        self.util = utils()
        
        
    def startCam(self):
        self.pca = PCA()
        try:
            self.cap = cv2.VideoCapture(0)
        except Exception as e:
            print('Cam Error : ', e)
        else:
            try:
                self.bThread = True
                self.thread = Thread(target=self.detection)
                self.thread.start()
            except Exception as e:
                print('Cam Error2 : ', e)
                      

    def stopCam(self):
        self.bThread = False

    def detection(self):
        time_count = 0
        time_list = []

        while self.bThread:
            ok, frame = self.cap.read()
            if not ok:
                print('cam read errror')
                break
            if(time_count<100):
                start = time.time()
                

            img = cv2.flip(frame,1) # 1 = 좌우반전 0 = 상하반전
            h, w, _ = img.shape
            # prepare input
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob) 

            # inference, find faces
            detections = self.net.forward()          
            rotate_img = img.copy()

            # postprocessing
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)

                    # draw rects
                    
                    # cv2.putText(img, '%.2f%%' % (confidence * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    d = dlib.rectangle(x1,y1,x2,y2)

                    shape = self.predictor(gray, d)
                    shape = face_utils.shape_to_np(shape)

                    crop_img = self.util.CropFace(image = rotate_img, eye_left=shape[36], eye_right=shape[45], shape= shape)

                    name = self.pca.face_test(crop_img)

                    if(name == "None"):
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), int(round(h/150)), cv2.LINE_AA)
                        cv2.putText(img, '%s'%name , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), int(round(h/150)), cv2.LINE_AA)
                        cv2.putText(img, '%s'%name , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            
            # create image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = QImage(img, w,h, QImage.Format_RGB888 ) 
            self.sendImage.emit(img)
            
            if(time_count<100):
                run_time = time.time() - start
                time_list.append(run_time)
                time_count += 1
                #print("time :",run_time )
            
            if(time_count == 100):
                time_arry = np.array(time_list)
                time_mean = time_arry.mean()
                #print("100 frame mean time :",time_mean )              
                #print("fps : %.1f"%(1./float(time_mean)))
                time_count += 1
            time.sleep(0.01)
        self.sendImage.emit(QImage())
        self.cap.release()
        print('thread finished')
    

    def startRegister(self,name):
        try:
            self.cap = cv2.VideoCapture(0)
            self.name = name
            self.prograss_run.connect(self.widget.progress)
        except Exception as e:
            print('Cam Error : ', e)
        else:
            try:
                self.bThread = True
                self.thread = Thread(target=self.Register)
                self.thread.start()
            except Exception as e:
                print('Cam Error2 : ', e)          


    def Register(self):
        
        # print("Register name : ", self.name)
        count_cropFace = -5
        # run_count=0
        
        while self.bThread and count_cropFace != 50:
            ok, frame = self.cap.read()
            if not ok:
                print('cam read errror')
                break
            
            img = cv2.flip(frame,1) # 1 = 좌우반전 0 = 상하반전
            h, w, _ = img.shape
            # prepare input
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob) 

            # inference, find faces
            detections = self.net.forward()          
            rotate_img = img.copy()

            count_face = 0
            # postprocessing
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    d = dlib.rectangle(x1,y1,x2,y2)
                
                    # draw rects
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), int(round(h/150)), cv2.LINE_AA)
                    cv2.putText(img, '%.2f%%' % (confidence * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if count_cropFace > -1:
                        cv2.putText(img,str(count_cropFace),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    count_face += 1
            
            if(count_face == 1):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                shape = self.predictor(gray, d)
                shape = face_utils.shape_to_np(shape)

                crop_img = self.util.CropFace(image = rotate_img, eye_left=shape[36], eye_right=shape[45], shape= shape)

                if not os.path.exists("faces/%s"%self.name):                   
                    os.makedirs("faces/"+self.name)   
                    
                file_name_path = 'faces/%s/'%self.name +str(count_cropFace)+'.jpg'
                if count_cropFace > -1:
                    cv2.imwrite(file_name_path,crop_img)
                count_cropFace += 1    

            # create image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = QImage(img, w, h, QImage.Format_RGB888 ) 
            self.sendImage.emit(img)

            
            time.sleep(0.01)
        # self.sendImage.emit(QImage())
        self.cap.release()
        if(count_cropFace == 50):
            self.prograss_run.emit()
        print('thread finished')

