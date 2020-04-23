#!/usr/bin/env bash
# curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b8fc5caa6503a8a66f4935c

echo -e "<device>"
curl -X POST -H "Content-Type: application/json" -d '
{
    "name":"pendulum_raspi_device",
    "description":"raspberry pi related to pendulum",
    "adminState":"UNLOCKED",
    "operatingState":"ENABLED",
    "addressable":{"name":"pendulum"},
    "labels":["pendulum","angle"],
    "location":"",
    "service":{"name":"raspi_device_service"},
    "profile":{"name":"pendulum_raspi_mqtt_device_profile"}
}
' http://localhost:48081/api/v1/device

echo -e ""

curl -X POST -H "Content-Type: application/json" -d '
{
    "name":"motor_raspi_device",
    "description":"raspberry pi related to motor",
    "adminState":"UNLOCKED",
    "operatingState":"ENABLED",
    "addressable":{"name":"motor"},
    "labels":["motor","angle"],
    "location":"",
    "service":{"name":"raspi_device_service"},
    "profile":{"name":"motor_raspi_mqtt_device_profile"}
}
' http://localhost:48081/api/v1/device

echo -e ""
echo -e ""