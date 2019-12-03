import cv2
import numpy as np

from utils import import_file, constants

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(constants['trained_folder'])
cascadePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

lines = import_file(constants['samples_map_file'])
mapper = {}

for line in lines:
    identifier, name = line[1].split(',')
    mapper[identifier] = name.replace('\n', '')

while True:
    ret, img =cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        text = str(mapper[str(Id)])
        pos = (x,y+h)
        thickness = 2
        color = (0,0,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1

        cv2.putText(img, text, pos, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('im',img)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()