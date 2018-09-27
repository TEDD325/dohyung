#!/usr/bin/env bash
# curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b8fc5caa6503a8a66f4935c

echo -e "<addressable>"

curl -X POST -H "Content-Type: application/json" -d '
{
    "name":"pendulum",
    "protocol":"TCP",
    "address":"192.168.137.10",
    "port":1883,
    "path":"/pendulum_angle",
    "publisher":"pendulum_addressable_publisher",
    "user":"link",
    "password":"0123",
    "topic":"none"
}
' http://localhost:48081/api/v1/addressable
echo -e ""

curl -X POST -H "Content-Type: application/json" -d '
{
    "name":"motor",
    "protocol":"TCP",
    "address":"192.168.137.10",
    "port":1883,
    "path":"/motor_power",
    "publisher":"motor_addressable_publisher",
    "user":"link",
    "password":"0123",
    "topic":"motor_power"
}
' http://localhost:48081/api/v1/addressable
echo -e ""
echo -e ""


# ..
#curl -X POST -H "Content-Type: application/json" -d '
#{
#    "name":"pendulum",
#    "protocol":"TCP",
#    "address":"192.168.137.10",
#    "port":1883,
#    "path":"/pendulum_angle",
#    "publisher":"pendulum_addressable_publisher",
#    "user":"link",
#    "password":"0123",
#    "topic":"none"
#}
#' http://localhost:48081/api/v1/addressable
#echo -e ""