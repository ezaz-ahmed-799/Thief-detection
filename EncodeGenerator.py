import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage



cred = credentials.Certificate("serviceAddKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facial-recognition-with-rt-db-default-rtdb.firebaseio.com/",
    'storageBucket':"facial-recognition-with-rt-db.appspot.com"
})

#importing the student images
folderpath = 'Images'
modepathlist = os.listdir(folderpath)
imglist = []
studentids=[]
for path in modepathlist:
    imglist.append(cv2.imread(os.path.join(folderpath,path)))
    studentids.append(os.path.splitext(path)[0])

    #print(path)
    #print(os.path.split(path)[0])

    filename = f'{folderpath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)


#print(studentids)


def findEncodings(imageslist):
    encodelist = []
    for img in imageslist:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist

print("Encoding Started...")
encodelistknown= findEncodings(imglist)
encodelistknownwithids=[encodelistknown,studentids]
print("Encoding Completed")

file =open("Encodefile.p",'wb')
pickle.dump(encodelistknownwithids, file)
file.close()