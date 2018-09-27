#!/usr/bin/env bash
# curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b8fc5caa6503a8a66f4935c

# echo -e "<device service>"
# curl -X POST -H "Content-Type: application/json" -d '
# {
#     "name":"raspi_device_service",
#     "description":"Manage Pendulum and Motor Angle Values",
#     "labels":["raspberry","MQTT"],
#     "adminState":"UNLOCKED",
#     "operatingState":"ENABLED",
#     "addressable":
#         {"name":"raspi_service_addressable"}
# }
# ' http://localhost:48081/api/v1/deviceservice
# echo -e ""
# echo -e ""


# service.name=raspi_device_service == "name":"raspi_device_service"