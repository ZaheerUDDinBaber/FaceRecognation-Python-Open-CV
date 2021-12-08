import cv2
import sqlite3
import numpy as np
from PIL import Image

#--------------------------------------------------------------------------------------------#
#-------------------------------INSERT OR UPDATE NEW RECORDE IN DATABASE---------------------#
def InsertOrUpdate(id,name,age,gen,type):
    conn = sqlite3.connect("FaceRec_Database.db")
    sqlCommand = "SELECT * from People WHERE ID =" +str(id)
    existance = conn.execute(sqlCommand)
    recordeExist = 0
    for row in existance:
        recordeExist = 1

    if (recordeExist==1):
        sqlCommand = "UPDATE People SET Name =" +str(name) + "WHERE ID =" +str(id)
    else:
        sqlCommand = "INSERT into People(ID,Name,Gender,Age,Type) VALUES ("+str(id)+","+str(name)+","+str(age)+","+str(gen)+","+str(type)+")"
        conn.execute(sqlCommand)
        conn.commit()
        conn.close()
# --------------------------------------------------------------------------------------------#

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #telling system what kind of face we are going to detect using that api of openCV
camera = cv2.VideoCapture(0) #defining object for camera feed

id = raw_input("enter user id : ") #uniqueid for each name
name = raw_input("enter user name : ")
age = raw_input("enter age : ")
gen = raw_input("enter gender : ")
type = raw_input("enter position : ")

InsertOrUpdate(id,name,age,gen,type)

sampleNumber = 0 #number of samples

while (True):
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #
    faces = faceDetect.detectMultiScale(gray, 1.3,5)
    for(x, y, w, h) in faces:
        sampleNumber = sampleNumber+1
        cv2.imwrite("dataset/user." + str(id) +"." + str(sampleNumber) + ".jpg", gray[y:y+h,x:x+w]) #
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #live image , coordinates to make rectangle over faces,RGB rectangle colour check value, thickness
        cv2.waitKey(100)

    cv2.imshow("Faces", img) #open camera feed frame name faces
    cv2.waitKey(1)

    if (sampleNumber>20): #rectrict samples
        break

camera.release() #destroying cam feed
cv2.destroyAllWindows() #exiting all frames
