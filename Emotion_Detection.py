import cv2
from deepface import DeepFace
trained=cv2.CascadeClassifier("C:\\Users\\Ponpr\\Downloads\\face (1).xml")
vid=cv2.VideoCapture(0)
   
while(True):
    ret,frame=vid.read()
    a=DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)
    b=str(a)
    if ret==True:
        faces=trained.detectMultiScale(frame)
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0,),2)
            cv2.putText(frame,b,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
cv2.waitKey()
cv2.destroyAllWindows()
