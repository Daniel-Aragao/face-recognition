import cv2, time
import numpy as np

from utils import import_file, constants

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(constants['trained_folder'])
cascadePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

thickness = 2
color = (0,0,255)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
time_out = 2

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

lines = import_file(constants['samples_map_file'])
mapper = {'0': 'No one'}

for line in lines:
    identifier, name = line[1].split(',')
    mapper[identifier] = name.replace('\n', '')

last_rec = 0
last_rec_time = time.time()


while True:
    ret, img =cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)

    if last_rec and type(faces) == type((0,0)):
        if time.time() - last_rec_time >= time_out:
            last_rec = 0
            last_rec_time = time.time()

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        text = str(mapper[str(Id)])
        pos = (x,y+h)

        cv2.putText(img, text, pos, font, fontScale, color, thickness, cv2.LINE_AA)

        last_rec = Id
        last_rec_time = time.time()

    cv2.imshow('im',img)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()