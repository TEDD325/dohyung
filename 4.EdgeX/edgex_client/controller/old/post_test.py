import requests
import json

headers = {"Content-Type" : "application/json"}
msg = {"motor_power": "10000000"}


r = requests.put(
            url="http://localhost:48082/api/v1/device/5b97f24e44d00aaaf3e8d413/command/5b97f11544d00aaaf3e8d40d",
            data=json.dumps(msg),
            headers=headers
)

print(r.status_code)