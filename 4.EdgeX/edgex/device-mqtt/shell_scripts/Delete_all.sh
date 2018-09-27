#!/usr/bin/env bash
# curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b8fc5caa6503a8a66f4935c

echo -e "<delete all>"


curl -X DELETE http://localhost:48071/api/v1/registration/id/5b97e94544d0eeb84d249f27
curl -X DELETE http://localhost:48071/api/v1/registration/id/5b97e94544d0eeb84d249f26
curl -X DELETE http://localhost:48081/api/v1/device/id/5b97e92c44d00aaaf3e8d407
curl -X DELETE http://localhost:48081/api/v1/device/id/5b97e92c44d00aaaf3e8d406
curl -X DELETE http://localhost:48081/api/v1/deviceservice/id/5b97e92844d00aaaf3e8d405
curl -X DELETE http://localhost:48081/api/v1/deviceprofile/id/5b97e91144d00aaaf3e8d3ff
curl -X DELETE http://localhost:48080/api/v1/valuedescriptor/id/5b97e8f944d0d780905e986c
curl -X DELETE http://localhost:48080/api/v1/valuedescriptor/id/5b97e8f944d0d780905e986b
curl -X DELETE http://localhost:48080/api/v1/valuedescriptor/id/5b97e8e144d0d780905e986a
curl -X DELETE http://localhost:48080/api/v1/valuedescriptor/id/5b97e8e144d0d780905e9869
curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b97e8c744d00aaaf3e8d3fd
curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b97e8c744d00aaaf3e8d3fc

echo -e ""
echo -e ""