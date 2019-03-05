#!/usr/bin/env bash
# curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b8fc5caa6503a8a66f4935c

 echo -e "<value descriptor>"
 curl -X POST -H "Content-Type: application/json" -d '
 {
     "name":"pendulum_angle",
     "description":"pendulum_angle_value",
     "min":"-180",
     "max":"180",
     "type":"I",
     "uomLabel":"angle",
     "defaultValue":"0",
     "formatting":"%s",
     "labels":["pendulum","angle"]
 }
 ' http://localhost:48080/api/v1/valuedescriptor
 echo -e ""

 curl -X POST -H "Content-Type: application/json" -d '
 {
     "name":"motor_angle",
     "description":"motor_angle_value",
    "min":"-36000",
    "max":"36000",
    "type":"I",
    "uomLabel":"angle",
    "defaultValue":"0",
    "formatting":"%s",
    "labels":["motor","angle"]
}
' http://localhost:48080/api/v1/valuedescriptor
echo -e ""

curl -X POST -H "Content-Type: application/json" -d '
{
    "name":"motor_power",
    "description":"motor_power_value",
    "min":"-100",
    "max":"100",
    "type":"I",
    "uomLabel":"power",
    "defaultValue":"0",
    "formatting":"%s",
    "labels":["motor","power"]
}
' http://localhost:48080/api/v1/valuedescriptor
echo -e ""

curl -X POST -H "Content-Type: application/json" -d '
{
    "name":"motor_status",
    "description":"motor_status_value",
    "min":"0",
    "max":"1",
    "type":"I",
    "uomLabel":"status",
    "defaultValue":"1",
    "formatting":"%s",
    "labels":["motor","status"]
}
' http://localhost:48080/api/v1/valuedescriptor
echo -e ""
echo -e ""