## addressable  <br>
- device service  <br>
POST to http://localhost:48081/api/v1/addressable  <br>
>   
    {
        "name":"raspi_service_addressable",
        "protocol":"TCP",
        "address":"localhost",
        "port":1883,
        "path":"/raspi_service",
        "publisher":"none",
        "user":"link",
        "password":"0123",
        "topic":"raspi_control"
    }

<code>5b9628ed9f8fc200012513ae</code>

- device1  <br>
POST to http://localhost:48081/api/v1/addressable  <br>
>
    {
        "name":"pendulum_addressable",
        "protocol":"TCP",
        "address":"localhost",
        "port":1883,
        "path":"/pendulum_addressable",
        "publisher":"none",
        "user":"link",
        "password":"0123",
        "topic":"pendulum_addressable"
    }

<code>5b9629419f8fc200012513af</code>
                       

- device2  <br>
POST to http://localhost:48081/api/v1/addressable  <br>
>
    {
        "name":"motor_addressable",
        "protocol":"TCP",
        "address":"localhost",
        "port":1883,
        "path":"/motor_addressable",
        "publisher":"none",
        "user":"link",
        "password":"0123",
        "topic":"motor_addressable"
    }

<code>5b9629629f8fc200012513b0</code>

------------------------------------------------

## valuedescriptor
- pendulum angle  <br>
POST to http://localhost:48080/api/v1/valuedescriptor  <br>
>
    {"name":"pendulum_angle_value",
    "description":"pendulum_angle_value", 
    "min":"-180",
    "max":"180",
    "type":"I",
    "uomLabel":"angle",
    "defaultValue":"0",
    "formatting":"%s",
    "labels":["pendulum","angle"]}

<code>5b962a1eda38f7992fa6c509</code>


- motor angle  <br>
POST to http://localhost:48080/api/v1/valuedescriptor  <br>
> 
    {"name":"motor_angle_value",
    "description":"motor_angle_value", 
    "min":"-36000",
    "max":"36000",
    "type":"I",
    "uomLabel":"angle",
    "defaultValue":"0",
    "formatting":"%s",
    "labels":["motor","angle"]}

<code>5b962a1eda38f7992fa6c509</code>

- motor power  <br>
POST to http://localhost:48080/api/v1/valuedescriptor  <br>
> 
    {"name":"motor_power_value",
    "description":"motor_power_value", 
    "min":"-100",
    "max":"100",
    "type":"I",
    "uomLabel":"angle",
    "defaultValue":"0",
    "formatting":"%s",
    "labels":["motor","power"]}

<code>5b962bf9da38f7992fa6c641</code>

- motor status  <br>
POST to http://localhost:48080/api/v1/valuedescriptor  <br>
> 
    {"name":"motor_status_value",
    "description":"motor_status_value", 
    "min":"0",
    "max":"1",
    "type":"I",
    "uomLabel":"angle",
    "defaultValue":"1",
    "formatting":"%s",
    "labels":["motor","status"]}

<code>5b962c20da38f7992fa6c65e</code>

- pendulum error  <br>
POST to http://localhost:48080/api/v1/valuedescriptor  <br>
> 
    {"name":"pendulum_error_value",
    "description":"pendulum_error_value", 
    "min":"",
    "max":"",
    "type":"S",
    "uomLabel":"",
    "defaultValue":"pendulum error",
    "formatting":"%s",
    "labels":["pendulum","error"]}

<code>5b962c80da38f7992fa6c6b0</code>

- motor error  <br>
POST to http://localhost:48080/api/v1/valuedescriptor  <br>
> 
    {"name":"motor_error_value",
    "description":"motor_error_value", 
    "min":"",
    "max":"",
    "type":"S",
    "uomLabel":"",
    "defaultValue":"motor error",
    "formatting":"%s",
    "labels":["motor","error"]}

<code>5b962cacda38f7992fa6c6d3</code>

------------------------------------------------

## device profile
POST to http://localhost:48081/api/v1/deviceprofile/uploadfile  <br>
![](https://docs.edgexfoundry.org/_images/EdgeX_WalkthroughPostmanFile.png)
>  
    name: "raspi_mqtt_device_profile"
    manufacturer: "LINKLAB"
    model: "Raspberry Pi 3"
    labels: 
        - "raspi"
    description: "raspi-mqtt-pendulum-device-profile"
    commands: 
      - 
        name: pendulumangle
        get: 
            path: "/api/v1/devices/{deviceId}/pendulumangle"
            responses:
              - 
                code: "200"
                description: "Angle of Pendulum"
                expectedValues: ["pendulum_angle_value"]
              -
                code: "503"
                description: "service unavailable"
                expectedValues: ["pendulum_error_value"]
      - 
        name: motorangle
        get: 
            path: "/api/v1/devices/{deviceId}/motorangle"
            responses:
              - 
                code: "200"
                description: "Angle of Motor"
                expectedValues: ["motor_angle_value"]
              -
                code: "503"
                description: "service unavailable"
                expectedValues: ["motor_error_value"]
      - 
        name: motorpower 
        put:
            path: "/api/v1/devices/{deviceId}/motorpower"
            parameterNames: ["motor_power_value"]
            responses:
              - 
                code: "204"
                description: "Set the power of motor."
                expectedValues: []
              -
                code: "503"
                description: "service unavailable"
                expectedValues: ["motor_error_value"]
      - 
        name: motorstatus
        put:
            path: "/api/v1/devices/{deviceId}/motorstatus"
            parameterNames: ["motor_status_value"]
            responses:
              - 
                code: "204"
                description: "Set the snapshot duration."
                expectedValues: []
              -
                code: "503"
                description: "service unavailable"
                expectedValues: ["motor_error_value"]
                
<code>5b962eed9f8fc200012513b5</code>

- if you want to delete one of the deviceprofile, first delete device related. 
------------------------------------------------

## device service
POST to http://localhost:48081/api/v1/deviceservice <br>
>
    {"name":"raspi_device_service",
    "description":"Manage Pendulum and Motor Angle Values",
    "labels":["raspberry","MQTT"],
    "adminState":"UNLOCKED",
    "operatingState":"ENABLED",
    "addressable":
        {"name":"raspi_service_addressable"}}
        
<code>5b9630a19f8fc200012513b6</code>

------------------------------------------------

## device
POST to http://localhost:48081/api/v1/device <br>
>
    {"name":"pendulum_raspi_device",
    "description":"raspberry pi related to pendulum",
    "adminState":"UNLOCKED",
    "operatingState":"ENABLED",
    "addressable":{"name":"pendulum_addressable"},
    "labels":["pendulum","angle"],
    "location":"",
    "service":{"name":"raspi_device_service"},
    "profile":{"name":"raspi_mqtt_device_profile"}}
    
    
    {"name":"motor_raspi_device",
    "description":"raspberry pi related to motor",
    "adminState":"UNLOCKED",
    "operatingState":"ENABLED",
    "addressable":{"name":"motor_addressable"},
    "labels":["motor","angle"],
    "location":"",
    "service":{"name":"raspi_device_service"},
    "profile":{"name":"raspi_mqtt_device_profile"}}
        

{"name":"countcamera1",
"description":"human and dog counting camera #1",
"adminState":"unlocked",
"operatingState":"enabled",
"addressable":{"name":"camera1 address"},
"labels":["camera","counter"],
"location":"",
"service":{"name":"camera control device service"},
"profile":{"name":"camera monitor profile"}}


<code>5b9637ba9f8fc200012513b9</code>
<code>5b9637d89f8fc200012513ba</code>

GET to http://localhost:48081/api/v1/deviceservice <br>
GET to http://localhost:48081/api/v1/deviceservice/label/pendulum <br>
GET to http://localhost:48081/api/v1/deviceservice/label/motor <br>

# command url
http://localhost:48082/api/v1/device/name/motor_raspi_device