import os
import pickle
import numpy
import cvzone
import cv2
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage



cred = credentials.Certificate("serviceAddKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facial-recognition-with-rt-db-default-rtdb.firebaseio.com/",
    'storageBucket':"facial-recognition-with-rt-db.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgbackground = cv2.imread('Resourses/background.png')

#importing the mode images into list
foldermodepath = 'Resourses/Modes'
modepathlist = os.listdir(foldermodepath)
imgmodelist = []
for path in modepathlist:
    imgmodelist.append(cv2.imread(os.path.join(foldermodepath,path)))


#Load the encoding file

print("Loading Encode File...")
file= open('Encodefile.p','rb')
encodelistknownwithids= pickle.load(file)
encodelistknown, studentids = encodelistknownwithids
#print(studentids)
print("Encode File Loaded")

modetype =0
counter=0
id=-1
imgstudent = []


while True:
    success, img = cap.read()

    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    facecurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs, facecurframe)

    imgbackground[162:162+480,55:55+640]= img
    imgbackground[44:44+633,808:808+414]= imgmodelist[modetype]

    for encodeface, faceloc in zip(encodecurframe, facecurframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        facedis = face_recognition.face_distance(encodelistknown,encodeface)
        #print("Matches",matches)
        #print("face distance",facedis)

        matchindex = np.argmin(facedis)
        #print("Match Index", matchindex)

        if matches[matchindex]:
            #print ("known face detected")
            #print(studentids[matchindex])
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 30+x1,120+y1,x2-x1,y2-y1
            cvzone.cornerRect(imgbackground, bbox, rt=0)
            id= studentids[matchindex]
            #print(id)

            if counter==0:
                counter=1
                modetype = 1

    if counter != 0:

        if counter == 1:
            studentinfo = db.reference(f'Suspects/{id}').get()
            #print(studentinfo)

            #Get the image from storage
            blob = bucket.get_blob(f'Images/{id}.jpg')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgstudent = cv2.imdecode(array,cv2.COLOR_BGRA2BGR)



        cv2.putText(imgbackground, str(studentinfo['total_attendance']),(861,125),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

        cv2.putText(imgbackground, str(studentinfo['crime']), (1018, 600),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
        cv2.putText(imgbackground, str(studentinfo['age']), (1018, 543),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

        #1:49 in the video

        cv2.putText(imgbackground, str(studentinfo['name']), (968, 400),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (50, 50, 50), 1)


        imgbackground[131:131+216,922:922+216] = imgstudent

        counter=+1





    cv2.imshow("Face recognition",imgbackground)
    if cv2.waitKey(10) == ord("e"):
        break
