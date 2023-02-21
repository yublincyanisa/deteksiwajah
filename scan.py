from ast import Import
import cv2
import time
import os
from PIL import Image
camera = 0
video = cv2.VideoCapture('rekam.mp4')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('DataSet/training.xml')
a = 0
while True:
    a = a+1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDetect.detectMultiScale(abu, 1.3, 5)
    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = recognizer.predict(abu[y:y+h, x:x+w])
        if (id == 1):
            id = 'Yublin'
        elif (id == 2):
            id = 'Hilda'

        cv2.putText(frame, str(id), (x+40, y-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
