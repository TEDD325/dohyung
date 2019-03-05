#!/usr/bin/env bash
# curl -X DELETE http://localhost:48081/api/v1/addressable/id/5b8fc5caa6503a8a66f4935c

echo -e "<device profile>"
curl \
  -F "file=@./PendulumProfile.yml" \
  http://localhost:48081/api/v1/deviceprofile/uploadfile
echo -e ""
curl \
  -F "file=@./MotorProfile.yml" \
  http://localhost:48081/api/v1/deviceprofile/uploadfile
echo -e ""
echo -e ""