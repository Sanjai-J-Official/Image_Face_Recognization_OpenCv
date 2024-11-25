import cv2 
import numpy as np 

haar_cascades=cv2.CascadeClassifier("haar_cascades.xml")
person=["tonnyStark","Vijay"]
# features=np.load("features.npy")
# labels=np.load("labels.npy")

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")

img=cv2.imread(r"C:\Users\anban\OneDrive\Pictures\dataset\045_caf1e891.jpg")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

 
face_rect=haar_cascades.detectMultiScale(gray , 1.1,4)

for (x,y,w,h) in face_rect:
    face_roi=gray[x:x+w,y:y+h]
    label,confidence=face_recognizer.predict(face_roi)

    print(f"{person[label]} with a confidence Score : {confidence}")

    cv2.putText(img,str(person[label]),(20,20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),thickness=2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=2)

cv2.imshow("Detected Image",img)

cv2.waitKey(0)
