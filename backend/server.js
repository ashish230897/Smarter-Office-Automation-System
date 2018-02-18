var express = require('express');
var fs = require('fs');
var util = require('util');
var app = express();
var http = require('http').Server(app);
var logger = require("./utils/logger");
var tempmqtt = require('mqtt')
var mqtt = require('./mqtt.js');

//var PythonShell = require('python-shell');
var time = new Date();
//For Implementing Sockets
//var io = require('socket.io')(http);
//var tmpSocket = require("./socket.js");
//tmpSocket.initializeSocket(io);

//*******************************firebase Admin setup**************************************//
var admin = require("firebase-admin");
// Fetch the service account key JSON file contents
var serviceAccount = require("./nvidiahack-d0e6d-firebase-adminsdk-ufe90-d92d50ff9b.json");

// Initialize the app with a service account, granting admin privileges
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://nvidiahack-d0e6d.firebaseio.com/"
});

// As an admin, the app has access to read and write all data, regardless of Security Rules
var db = admin.database();
var FCM = require('fcm-node');
var serverKey = 'AAAAeyjakNQ:APA91bG0ueYwIO9B2WwMB0HF1Y4LGACLvlMUJwCkj9Ac9QuJK72268G8RFosoGz376MYfoYFXf3RR4zsvBCZJY0n6PIdFpI51-pZh_BONWG3YFwSDGI2JcsDuNxkj4_3DvFgM-iIuWb3'; //put your server key here
var fcm = new FCM(serverKey);
var utils;
var client = mqtt.mqttConnect(tempmqtt);

var enterTime ='9:00';
var leavingTime ='17:59';


app.get('/', function(req, res) {

    console.log('requestmade')

    res.sendfile('./index.html');
});

app.post('/lightNumber',function(req,res){

	 req.on('data', function(chunk) {
      var body=JSON.parse(chunk);
      db.ref("/officeApliances").once("value",function(snapshot){
          console.log(snapshot.val());
        });
      db.ref("/officeApliances").update({
          "ac": body.ac,
	  "light1":body.light1
        });

     console.log(chunk);
   });
   res.send('Successfully Sent');
 });

app.post('/arrival/userAttendance',function(req,res){
    req.on('data',function(chunk){
      var time = new Date();
      var currentTime  = "" + time.getHours() + ":" + time.getMinutes() + "";
      var currentYear  =  time.getFullYear();
      var currentMonth =  time.getMonth();
      var currentDay   =  time.getDate();
      console.log(currentTime);
      console.log(currentYear);
      console.log(currentMonth);
      console.log(currentDay);

      if (currentTime < enterTime){
        admin.auth().getUserByEmail("nvidia@gmail.com")
        .then(function(userRecord) {
      // See the UserRecord reference doc for the contents of userRecord.
            var userRecordtmp = userRecord.toJSON();
            //console.log("Successfully fetched user data:", userRecordtmp.uid);

            var uid = userRecordtmp.uid;
            db.ref("/"+ uid +"/attendance"+"/"+currentYear+"/"+currentMonth+"/"+currentDay+"").set({
               "Time": ""+currentTime+"",
               "late": true
             });
      })
      .catch(function(error) {
        console.log("Error fetching user data:", error);
      });
    }
    res.send('Successfully Sent');
  });
});
app.post('/departure/userAttendance',function(req,res){
    req.on('data',function(chunk){
      var time = new Date();
      var currentTime  = "" + time.getHours() + ":" + time.getMinutes() + "";
      var currentYear  =  time.getFullYear();
      var currentMonth =  time.getMonth();
      var currentDay   =  time.getDate();
      console.log(currentTime);
      console.log(currentYear);
      console.log(currentMonth);
      console.log(currentDay);

      if (currentTime < leavingTime){
        admin.auth().getUserByEmail("nvidia@gmail.com")
        .then(function(userRecord) {
      // See the UserRecord reference doc for the contents of userRecord.
            var userRecordtmp = userRecord.toJSON();
            console.log("Successfully fetched user data:", userRecordtmp.uid);

            var uid = userRecordtmp.uid;
            db.ref("/"+ uid +"/attendance"+"/"+currentYear+"/"+currentMonth+"/"+currentDay+"").set({
               "Time": ""+currentTime+"",
               "late": true
             });

      })
      .catch(function(error) {
        console.log("Error fetching user data:", error);
      });
    }
    res.send('Successfully Sent');
  });
});

db.ref("/objectScanCount").on("value", function(snapshot) {
            var body =  snapshot.val()
            admin.auth().getUserByEmail("nvidia@gmail.com")
                  .then(function(userRecord) {
          // See the UserRecord reference doc for the contents of userRecord.
            var userRecordtmp = userRecord.toJSON();
      //console.log("Successfully fetched user data:", userRecordtmp.uid);
            var uid = userRecordtmp.uid;
            var	chunk = '{"topic":"/newInventaryRequests","message":"'+body.count+'"}';
            console.log(chunk);
             mqtt.mqtt_pub(client,chunk);
          })
          .catch(function(error) {
            console.log("Error fetching user data:", error);
          });
      });

http.listen(443,function(){
	logger.info("Signallng server is listening on port 443");
});
