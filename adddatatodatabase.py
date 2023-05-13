import firebase_admin
from firebase_admin import db
from firebase_admin import credentials

cred = credentials.Certificate("serviceAddKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facial-recognition-with-rt-db-default-rtdb.firebaseio.com/"
})


ref =db.reference('Suspects')

data= {
    "4241":
        {
            "name": "EZAZ",
            "crime": "Stealing",
            "age":"19",
            "total_attendance":10

        },
    "4242":
        {
            "name": "Deepika",
            "crime": "Beautiful",
            "age": "37",
            "total_attendance":7

        },
    "4243":
        {
            "name": "Elon",
            "crime": "Smart",
            "age": "51",
            "total_attendance":9

        }
}
for key,value in data.items():
    ref.child(key).set(value)