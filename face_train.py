import os
import cv2
import numpy as np
people=['tonnystark', 'vijay']
 

DIR=r"C:\Users\anban\OneDrive\Documents\image predict\dataset"

features=[]
labels=[]
haar_cascades=cv2.CascadeClassifier(r"C:\Users\anban\OneDrive\Desktop\azure project\opencv\haar_cascades.xml")


def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)
        
        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv2.imread(img_path)
            gray=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

            face_rect=haar_cascades.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)      
 
            for (x,y,w,h) in face_rect:
                face_roi=gray[y:y+h,x:x+w]
                features.append(face_roi)
                labels.append(label)
            
            
create_train()  
print("Training Done-----------------")
labels=np.array(labels)
print("labels convert to np array completed---------")
features=np.array(features,dtype="object")


print("features convert to Np array completed---------")
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
print("face recognizer created---------")
face_recognizer.train(features,labels)
print("face recognizer training completed---------")
face_recognizer.save("face_recognizer.yml")
print("face recognizer completed---------")
np.save("features.npy",features)
np.save("labels.npy",labels)
print("Files Saved--------------")
