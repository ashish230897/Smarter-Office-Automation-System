#include <ESP8266WiFi.h>
#include <FirebaseArduino.h>

// Set these to run example.
#define FIREBASE_HOST "nvidiahack-d0e6d.firebaseio.com"         //Firebase Console -> SelectProject -> Database ->Link Name without Https
#define FIREBASE_AUTH "xTnfYQQhGWwOEfFonotdwkb3zFr5MbVLcu80qhaX" ///Firebase Console -> SelectProject -> Got o ProjectSettings -> Service accounts -> databse Secrets -> Secret Key
#define WIFI_SSID "L&T-IIoT"                                       //Name of your network
#define WIFI_PASSWORD "realtime10"                            // WIFI_PASSWORD

void setup() {
  Serial.begin(9600);
  // connect to wifi.
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("connecting");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  Serial.print("connected: ");
  Serial.println(WiFi.localIP());
  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);
  Serial.println("Connection to firebase databse established");
  pinMode(12, OUTPUT);// Initialize the LED_BUILTIN pin as an output
  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);
  digitalWrite(12, LOW);
}


void loop() {
  // put your main code here, to run repeatedly:
//Serial.print(Firebase.getBool("officeApliances/light1"));
//delay(500);
//Firebase.setBool("officeApliances/light1",0);
//if (Firebase.failed()) {
//     Serial.println(Firebase.error());
//     return;
// }
// 
// Serial.print(Firebase.getBool("officeApliances/light1"));
//Firebase.setBool("officeApliances/light1",1);
//if (Firebase.failed()) {
//     Serial.println(Firebase.error());
//     return;
//}

 if(Firebase.getBool("officeApliances/ac")){
  digitalWrite(12, HIGH);// Turn the LED off by making the voltage HIGH
  digitalWrite(13, HIGH);
 }
 else{
  digitalWrite(12, LOW);   // Turn the LED on (Note that LOW is the voltage level
  digitalWrite(13, LOW);             // but actually the LED is on; this is because 
 }
 

}
