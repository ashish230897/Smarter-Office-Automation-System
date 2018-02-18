
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import requests as req
person_count=9
cred = credentials.Certificate('nvidiahack-d0e6d-firebase-adminsdk-ufe90-d92d50ff9b.json')
firebase_admin.initialize_app(cred,{ 'databaseURL':'https://nvidiahack-d0e6d.firebaseio.com'})
ref = db.reference('/')
print(ref.get())
db.reference('/Heatmap').update({ 'numberOfPeople': person_count})
