import math
import numpy as np
import cv2, os
import pandas as pd
from numpy import linalg as LA
from PIL import Image

class utils:
    def __init__(self):
        super().__init__()
    
    def ImageRotate(self, image, angle, center = None, resample=Image.BICUBIC):
        cx, cy = center
        cosine = math.cos(angle)
        sine = math.sin(angle)
        a = cosine
        b = sine
        c = cx - cx*a - cy*b
        d = -sine
        e = cosine
        f = cy - cx*d - cy*e
        return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

    def Rotate(self, a, b, c, d, angle, center):
        cx, cy = center
        cosine = math.cos(angle)
        sine = math.sin(angle)
        
        a1=np.zeros((2,))
        b1=np.zeros((2,))
        c1=np.zeros((2,))
        d1=np.zeros((2,))
        
        a1[0] = cosine*(a[0]-cx)-sine*(a[1]-cy)+cx
        a1[1] = sine*(a[0]-cx)+cosine*(a[1]-cy)+cy

        b1[0] = cosine*(b[0]-cx)-sine*(b[1]-cy)+cx
        b1[1] = sine*(b[0]-cx)+cosine*(b[1]-cy)+cy

        c1[0] = cosine*(c[0]-cx)-sine*(c[1]-cy)+cx
        c1[1] = sine*(c[0]-cx)+cosine*(c[1]-cy)+cy

        d1[0] = cosine*(d[0]-cx)-sine*(d[1]-cy)+cx
        d1[1] = sine*(d[0]-cx)+cosine*(d[1]-cy)+cy
        
        return a1[0], b1[1], c1[0], d1[1]

    def CropFace(self, image, eye_left=(0,0), eye_right=(0,0),shape = np.array):
        image = Image.fromarray(image, 'RGB')

        # get the direction
        eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])

        rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))

        # rotate original around the left eye
        image = self.ImageRotate(image, center=eye_left, angle=rotation)

        a=shape[0]
        b=shape[8]
        c=shape[16]
        d=shape[24]
        
        p, q, r, s = self.Rotate(a, b, c, d, angle=rotation, center = eye_left)
        
        # crop the rotated image
        image = image.crop((int(p), int(s), int(r), int(q)))
        
        # resize it
        image = image.resize((150, 150), Image.ANTIALIAS)
        
        return np.array(image)

class PCA:
    def __init__(self):
        super().__init__()
        # meanface, eigenvector 
        FaceDB_info = pd.read_csv("./db/FaceDB_info.csv", index_col=0)
        FaceDB_info = np.array(FaceDB_info)
        self.mean_face = FaceDB_info[:, 0].reshape(-1,1)
        self.eigenvector = FaceDB_info[:, 1:]

        # SID_weight, SID_index 
        SID_weight = pd.read_csv("./db/FaceDB_SID.csv", index_col=0)
        self.SID_index = list(SID_weight.index)
        self.SID_weight = np.array(SID_weight)    

    def face_db_raw_register(self,S, y):
        FaceDB = pd.concat([pd.DataFrame(data={"SID":y}), pd.DataFrame(S.T)], axis=1)
        FaceDB.to_csv("FaceDB_all_pictures_raw.csv", mode='w')
        
    def face_db_raw_load(self):
        raw_pictures = pd.read_csv("FaceDB_all_pictures_raw.csv", index_col=0)
        raw_pictures = np.array(raw_pictures)
        pictures = raw_pictures[:, 1:].T
        y = list(raw_pictures[:, 0])
        return pictures, y
        
    def face_matrix(self):
        S = []
        Y = []
        
        for name in os.listdir("./faces"):
            nameList = os.listdir("./faces/%s"%name)

            if len(nameList) <50 : 
                continue
                
            y = name.replace(".jpg","")
           
            count = 0
            for i in nameList:
                if count == 50 :
                    break
                filePath = "./faces/%s/%s"%(name,i)
                
                # decoding path written by korean
                stream = open( filePath.encode("utf-8") , "rb")
                bytes = bytearray(stream.read())
                numpyArray = np.asarray(bytes, dtype=np.uint8)
                img = cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgArr = np.array(gray)
                temp = np.reshape(imgArr, (150*150))
                Y.append(y)
                S.append(temp)
                count += 1
       
        S = np.array(S).T
        return S , Y
    
    def histo_equalization(self, S):
        SS = np.zeros_like(S)
        
        for i in range(S.shape[1]):
            SS[:,i] = cv2.equalizeHist(S[:,i]).reshape(-1,)
        
        m = np.mean(SS, axis=1)
        m = m.reshape(-1, 1)
        D = SS-m
        return D, m

    def eigenface(self,D, eigenface_num=20):
        L = D.T @ D

        # w : eigenvalue, v : eigenvector
        w, v = LA.eig(L) 
        
        # remove zero eignvalue
        zero_eigen = [i for i in range(len(w)) if abs(w[i]) <1e-4]
        #zero_eigen = np.where(abs(w)<1e-4)[0]
        
        j=0
        for i in zero_eigen:
            w = np.delete(w, i-j, axis=0)
            v = np.delete(v, i-j, axis=1)
            j += 1

        # sort eigenvalue
        index = np.argsort(w.real)[::-1] 
        ww = w.real[index]
        vv = v.real[:,index] 
        
        ww = ww[:eigenface_num]
        vv = vv[:, :eigenface_num]

        u = D @ vv
        u = u/(LA.norm(u, axis=0))
        
        weight = u.T @ D 
        return u, weight

    def face_db_register(self,y, weight):
        if len(y)!=weight.shape[1]:
            return ("face_db에 저장할 수 없습니다.")

        FaceDB = pd.concat([pd.DataFrame(data={"SID":y}),pd.DataFrame(weight.T)], axis=1) 
        FaceDB.to_csv("./db/FaceDB_all_pictures.csv", mode='w')
        
        FaceDB_person = FaceDB.groupby(["SID"]).mean() 
        FaceDB_person.to_csv("./db/FaceDB_SID.csv", mode='w')

    def face_db_info_register(self,mean_face, eigenvector):
        mean_face_df = pd.DataFrame(data=mean_face, columns=["mean_face"])
        eigen_df = pd.DataFrame(data=eigenvector, columns=["eigenvector_"+str(i+1) for i in range(eigenvector.shape[1])])
        FaceDB_info = pd.concat([mean_face_df, eigen_df], axis=1)
        FaceDB_info.to_csv("./db/FaceDB_info.csv", mode='w')


    def Euclidean_recognition(self, img, mean_face, eigenvector, SID_weight, SID_index, threshold=5000):
        img = cv2.equalizeHist(img) - mean_face
        img_weight = np.dot(img.T,eigenvector)
     
        weight_diff = [np.linalg.norm(img_weight-SID_weight[i]) for i in range(SID_weight.shape[0])]   

        SID = SID_index[np.argmin(weight_diff)] 

        if np.min(weight_diff)>threshold:          
            return "None"
        return SID

    
    def cosine_recognition(self, img, threshold=50):
        img = cv2.equalizeHist(img) - self.mean_face
        img_weight = np.dot(img.T, self.eigenvector)
        # dist1 = np.sqrt(np.sum(img_weight*img_weight)) 
        dist1 = LA.norm(img_weight)
        cosine_list = []
        for i in range(self.SID_weight.shape[0]):
            #dist2 = np.sqrt(np.sum(self.SID_weight[i]*self.SID_weight[i]))
            dist2 = LA.norm(self.SID_weight[i])
            cosine_similiarity = np.dot(img_weight, self.SID_weight[i])/(dist1 * dist2)
            cosine_list.append(cosine_similiarity)


        if (100*np.max(cosine_list))<threshold:          
            return "None"
        
        SID = self.SID_index[np.argmax(cosine_list)]
        return SID

    def face_register(self):  
        S, y = self.face_matrix() 
        D, mean_face = self.histo_equalization(S)
        eigenvector, weight = self.eigenface(D,20)
        self.face_db_register(y, weight)
        self.face_db_info_register(mean_face, eigenvector)

    def face_test(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        SID = self.cosine_recognition(gray.reshape(-1))
        
        return SID