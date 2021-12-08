import os
import cv2
import sqlite3
import numpy as np
from PIL import Image


faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
recognizer =cv2.face.LBPHFaceRecognizer_create()

recognizer.read('dataset\\trainingdata.xml') #load trained xml on rec..

#-----------------------------------------------------------------------#
def getProfile(id):
    conn = sqlite3.connect("FaceRec_Database.db")
    sqlCommand = "SELECT * from People WHERE ID="+str(id)
    commandExecute = conn.execute(sqlCommand)

    profile = None #null inatializer

    for row in commandExecute:
        profile = row
    conn.close() #closing the connection
    return profile
#-----------------------------------------------------------------------#

font = cv2.FONT_HERSHEY_SIMPLEX #font using on triangle
id = 0

while (True):
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        profile = getProfile(id)
        if(profile!=None):
           cv2.putText(img, str(profile[1]),(h+20,w+10), font, 1, (255, 255, 255), 2, cv2.LINE_AA) #profile[1] indicates the db table object location 1 which is id
           cv2.putText(img, str(profile[2]),(h+20,w+30), font, 1, (255, 255, 255), 2, cv2.LINE_AA) #profile[2] indicates the db table object location 2 which is name
           cv2.putText(img, str(profile[3]),(h+20,w+60), font, 1, (255, 255, 255), 2, cv2.LINE_AA) #profile[3] indicates the db table object location 1 which is age
           cv2.putText(img, str(profile[4]),(h+20,w+90), font, 1, (255, 255, 255), 2, cv2.LINE_AA) #profile[4] indicates the db table object location 1 which is position


    cv2.imshow("Faces", img)
    if( cv2.waitKey(1) == ord('q') ):
        break

camera.release()
cv2.destroyAllWindows()
