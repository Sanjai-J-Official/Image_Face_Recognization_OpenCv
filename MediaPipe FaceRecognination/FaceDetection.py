import cv2
import mediapipe as mp
import time


cap=cv2.VideoCapture("DemoVideo/2.mp4")

class FaceDetect():
    def __init__(self):
        self.mpfacedetetction=mp.solutions.face_detection
        self.facedetect=self.mpfacedetetction.FaceDetection()
        self.mpdraw=mp.solutions.drawing_utils
         
    def faceDetection(self,img):
        

        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.facedetect.process(imgrgb)

        if self.result.detections:
            
            for id, detect in  enumerate(self.result.detections):
                #mpdraw.draw_detection(img,detect)
                #print(detect)
                bboxc=detect.location_data.relative_bounding_box
                h,w=img.shape[:2]

                bbox=int(bboxc.xmin*w),int(bboxc.ymin*h),int(bboxc.width*w),int(bboxc.height*h)
                
                cv2.rectangle(img,bbox,(0,0,255),4)
                cv2.putText(img,f"{int(detect.score[0]*100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(222,0,0),4)
        return img

classface=FaceDetect()

def main():
    ptime=0
    while True:
        success,imgF=cap.read()
        if not success:
            break
    #resize Img----------------
        scale=0.40
        h=int(imgF.shape[1]*scale)
        w=int(imgF.shape[0]*scale)
        img=cv2.resize(imgF,(h,w))
    #face Detect Process cvt RGB
        img=classface.faceDetection(img)

    #fps 
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime


        cv2.putText(img,f"Fps:{int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,2,(99,25,5),3)
        cv2.imshow("img",img)

    #video Stop---------------
        if cv2.waitKey(10) & 0xFF==ord("q"):
            break

if __name__=="__main__":
    main()
