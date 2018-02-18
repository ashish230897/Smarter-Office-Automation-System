
exports.mqttConnect = function(mqtt){
    var client  = mqtt.connect('mqtt://13.126.45.185')//Default port is 1883 and enter the correct ip of the broker
    client.on('connect', function () {
        console.log("Connected to remote broker successfully")
        })
        return client
}

exports.mqtt_pub = function(client,payload){
var body = JSON.parse(payload)
    client.publish(body.topic,body.message)
    console.log("'"+body.topic+"'"+","+ "'"+body.message+"'");
}

exports.mqtt_sub = function(client,topic){
    client.subscribe(topic)
    console.log("Successfully subscribed to topic '"+topic+"'");
}

exports.payload = function(client){

    client.on('message', function (topic, message) {
      // message is Buffer
      console.log(topic.toString()+':'+message.toString())
//Depending on the message arrived the following function can pubish or perform certion functions or operation

      if(topic.toString() == '/light/intensity')
      {
          console.log('New value :'+ message.toString());
      }
      if(topic.toString() == '/lights')
      {
          if(message.toString()=='on'){
              mqtt_pub(client,'{"topic":"/light/intensity","message":"100"}');
              console.log('Light 1 is switched ON');
          }
          if(message.toString()=='off'){
              mqtt_pub(client,'{"topic":"/light/intensity","message":"0"}');
              console.log('Light 1 is switched OFF');
          }
      }

     client.end()
      console.log('After return')
    })
}
