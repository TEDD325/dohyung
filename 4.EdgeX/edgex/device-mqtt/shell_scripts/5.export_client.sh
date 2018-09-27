#!/usr/bin/env bash
# curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b8fc5caa6503a8a66f4935c
# http://localhost:48071/api/v1/registration

echo -e "<export client>"
curl -X POST -H "Content-Type: application/json" -d '
{
"name":"both-angle-export-client",
"addressable": {
	"name":"MosquittoMQTTBroker",
	"protocol":"TCP",
	"address":"192.168.137.10",
	"port":1883,
	"publisher":"pendulum_export_client",
	"user":"link",
	"password":"0123",
	"topic":"state_info_from_edgex"
	},
"format":"JSON",
"enable":true,
"filter": {
    "valueDescriptorIdentifiers":["pendulum_angle", "motor_angle"]
},
"destination":"MQTT_TOPIC"
}
' http://localhost:48071/api/v1/registration
echo -e ""

####### remove
curl -X POST -H "Content-Type: application/json" -d '
{
"name":"motor-power-response-status-export-client",
"addressable": {
	"name":"MosquittoMQTTBroker2",
	"protocol":"TCP",
	"address":"192.168.137.10",
	"port":1883,
	"publisher":"motor_export_client",
	"user":"link",
	"password":"0123",
	"topic":"motor_power_response_from_edgex"
	},
"format":"JSON",
"enable":true,
"filter": {
    "valueDescriptorIdentifiers":["motor_status"]
},
"destination":"MQTT_TOPIC"
}
' http://localhost:48071/api/v1/registration
echo -e ""




