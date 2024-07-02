from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import keyboard


faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotionModel = load_model('emocionalniModel.h5')
labels= [' Angry ','Disgust ','Fear','Happy','Neutral','Sad','Surprise']
cap=cv2.VideoCapture(0)
frameSkip=2
frameCount= 0

try:
    while True:
        ret,frame= cap.read()
        if not ret:
            print("nema")
            break
        frameCount +=1
        if frameCount%frameSkip!= 0:
            continue
        grayFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces =faceClassifier.detectMultiScale(grayFrame, 1.2,7)

        for (x,y, w,h) in faces:
            faceGray=grayFrame[y:y+h, x:x+w]
            faceGray= cv2.resize(faceGray,(48,48), interpolation=cv2.INTER_AREA)
            face =faceGray.astype('float')/ 255.0
            face= img_to_array(face)
            face =np.expand_dims(face,axis=0)
            predictions =emotionModel.predict(face)[0]
            label =labels[predictions.argmax()]
            labelPosition= (x,y)
            cv2.putText(frame, label,labelPosition,cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
            for i in range(len(labels)):
                emotion =labels[i]
                prob=predictions[i]
                text= f"{emotion}: {prob:.2f}"
                cv2.putText(frame,text,(10,(i+1)*20), cv2.FONT_HERSHEY_COMPLEX,0.5, (0,0,0), 1)

        cv2.imshow('',frame)
        if cv2.waitKey(1)==ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
