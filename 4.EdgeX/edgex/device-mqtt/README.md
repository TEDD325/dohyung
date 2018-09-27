Main Author:  Jim White

Copyright 2016-17, Dell, Inc.

MQTT Micro Service - device service for connecting an MQTT topic to EdgeX acting like a device/sensor feed.


# REST Endpoint
meta.db.addressable.url=http://localhost:48081/api/v1/addressable
meta.db.deviceservice.url=http://localhost:48081/api/v1/deviceservice
meta.db.deviceprofile.url=http://localhost:48081/api/v1/deviceprofile
meta.db.device.url=http://localhost:48081/api/v1/device
meta.db.devicereport.url=http://localhost:48081/api/v1/devicereport
meta.db.command.url=http://localhost:48081/api/v1/command
meta.db.event.url=http://localhost:48081/api/v1/event
meta.db.schedule.url=http://localhost:48081/api/v1/schedule
meta.db.provisionwatcher.url=http://localhost:48081/api/v1/provisionwatcher
meta.db.ping.url=http://localhost:48081/api/v1/ping

http://localhost:48080/api/v1/event/device/RASPI/100


# Setting
1. Addressable
- id: 5b90f20e9f8fc200018cc88f
- name: raspi-mqtt-pendulum
- post (create): http://localhost:48081/api/v1/addressable
{
        "name": "raspi-mqtt-pendulum",
        "protocol": "TCP",
        "address": "192.168.137.3",
        "port": 49999,
        "path": "/raspi_mqtt_pendulum",
        "publisher": "none",
        "user": "none",
        "password": "none",
        "topic": "none"
}

-id: 5b90f23d9f8fc200018cc890
- name: raspi-mqtt-motor
- post (create): http://localhost:48081/api/v1/addressable
{
        "name": "raspi-mqtt-motor",
        "protocol": "TCP",
        "address": "192.168.137.11",
        "port": 49999,
        "path": "/raspi_mqtt_motor",
        "publisher": "none",
        "user": "none",
        "password": "none",
        "topic": "none"
}


2. device profile

pendulum device profile
- id: 5b90ecf79f8fc200018cc88c
- name: raspi-mqtt-pendulum-device-profile

motor device profile
- id: 5b90ed2b9f8fc200018cc88e
- name: raspi-mqtt-motor-device-profile


3. device service 
- id: 5b7581c29f8fc20001fe0b53
- name: raspi-mqtt-service


4. device
0) All devices
- GET
http://localhost:48082/api/v1/device

1) Pendulum RASPI
- id: 5b90f3559f8fc200018cc891
- name: raspi-mqtt-pendulum-device
- post (create): http://localhost:48081/api/v1/device
{
	"name":"raspi-mqtt-pendulum-device",
	"description":"raspi-mqtt-pendulum-device",
	"adminState":"UNLOCKED",
	"operatingState":"ENABLED",
	"addressable": {"name":"raspi-mqtt-pendulum"},
	"labels": ["raspi","mqtt","pendulum", "device"],
	"service":{"name":"raspi-mqtt-service"},
	"profile":{"name":"raspi-mqtt-pendulum-device-profile"}
}

2) Motor RASPI
- id: 5b92b0469f8fc200014b5ad3
- name: raspi-mqtt-motor-device
- GET
http://localhost:48082/api/v1/device/name/raspi-mqtt-motor-device
http://localhost:48081/api/v1/device/profilename/raspi-mqtt-motor-device-profile

- POST
http://localhost:48081/api/v1/device
{
	"name":"raspi-mqtt-motor-device",
	"description":"raspi-mqtt-motor-device",
	"adminState":"UNLOCKED",
	"operatingState":"ENABLED",
	"addressable": {"name":"raspi-mqtt-motor"},
	"labels": ["raspi","mqtt","motor", "device"],
	"service":{"name":"raspi-mqtt-service"},
	"profile":{"name":"raspi-mqtt-motor-device-profile"}
}

5. Export Client

1) Pendulum/Motor Angle: Edgex --> Environment (Client)
- id: 5b913276fd4ca8000135bf5d

- POST
http://localhost:48071/api/v1/registration
{
"name":"state-info-export-client",
"addressable": {
	"name":"MosquittoMQTTBroker",
	"protocol":"TCP",
	"address":"192.168.137.10",
	"port":1883,
	"publisher":"link",
	"user":"link",
	"password":"0123",
	"topic":"state_info_from_edgex"
	},
"format":"JSON",
"enable":true,
"filter": {
    "valueDescriptorIdentifiers":["pendulum_angle_value", "motor_angle_value"]
},
"destination":"MQTT_TOPIC"
}


2) Motor Power Setting Response Status: Edgex --> Environment (Client)

- POST
http://localhost:48071/api/v1/registration
{eviceprofile
"name":"motor-power-response-status-export-client",
"addressable": {
	"name":"MosquittoMQTTBroker",
	"protocol":"TCP",
	"address":"192.168.137.10",
	"port":1883,
	"publisher":"link",
	"user":"link",
	"password":"0123",
	"topic":"motor_power_response"
	},
"format":"JSON",
"enable":true,
"filter": {
    "valueDescriptorIdentifiers":["motor_status"]
},
"destination":"MQTT_TOPIC"
}


http://localhost:49982/api/v1/device/5b911f9b9f8fc200018cc897/command/5b911e069f8fc200018cc894




# command url
http://localhost:48082/api/v1/device/name/motor_raspi_device





Device [name=motor_raspi_device, adminState=UNLOCKED, operatingState=ENABLED, addressable=Addressable [name=motor_addressable, protocol=TCP, address=192.168.0.10, port=1883, path=/motor_addressable, publisher=link, user=link, password=0123, topic=motor_addressable, toString()=BaseObject [id=5b9687ab44d0a85d011fb431, created=1536591787747, modified=1536591787747, origin=0]], lastConnected=1536655618275, lastReported=0, labels=[motor, angle], location=, service=DeviceService [adminState=UNLOCKED, operatingState=ENABLED, addressable=Addressable [name=raspi_service_addressable, protocol=HTTP, address=localhost, port=49982, path=/raspi_service, publisher=link, user=link, password=0123, topic=raspi_control, toString()=BaseObject [id=5b9687ab44d0a85d011fb42f, created=1536591787697, modified=1536652649430, origin=0]]], profile=DeviceProfile [name=motor_raspi_mqtt_device_profile, manufacturer=LINKLAB, model=Raspberry Pi 3, labels=[motor], objects=[DeviceObject [name=motor_angle, tag=null, description=RASPI motor angle degree values, properties=ProfileProperty [value=PropertyValue{readWrite:R, minimum:-36000, maximum:36000, defaultValue:0, size:6, precision:null, word:2, LSB:null, mask:0x00, shift:0, scale:1.0, offset:0.0, base:0, assertion:null, signed:true}, units=org.edgexfoundry.domain.meta.Units@7f7427d8], attributes={name=motor_angle}], DeviceObject [name=motor_power, tag=null, description=RASPI motor power values, properties=ProfileProperty [value=PropertyValue{readWrite:W, minimum:-100, maximum:100, defaultValue:0, size:3, precision:null, word:2, LSB:null, mask:0x00, shift:0, scale:1.0, offset:0.0, base:0, assertion:null, signed:true}, units=org.edgexfoundry.domain.meta.Units@64f3259a], attributes={name=motor_power}], DeviceObject [name=motor_status, tag=null, description=RASPI motor power set status, properties=ProfileProperty [value=PropertyValue{readWrite:R, minimum:0, maximum:1, defaultValue:1, size:1, precision:null, word:2, LSB:null, mask:0x00, shift:0, scale:1.0, offset:0.0, base:0, assertion:null, signed:true}, units=org.edgexfoundry.domain.meta.Units@781af7c], attributes={name=motor_status}]], commands=[Command [name=motor_angle, get=Action [path=/api/v1/device/{deviceId}/motorangle, responses=[Response [code=200, description=Angle of Motor, expectedValues=[motor_angle]], Response [code=503, description=service unavailable, expectedValues=[]]]], put=null, BaseObject [id=5b977d6044d07f79eaccf700, created=1536654688104, modified=1536654688104, origin=0]], Command [name=motor_power, get=null, put=Put [parameterNames=[motor_power]], BaseObject [id=5b977d6044d07f79eaccf701, created=1536654688105, modified=1536654688105, origin=0]], Command [name=motor_status, get=Action [path=/api/v1/device/{deviceId}/motorpoweridx, responses=[Response [code=200, description=Set the snapshot duration., expectedValues=[]], Response [code=503, description=service unavailable, expectedValues=[]]]], put=null, BaseObject [id=5b977d6044d07f79eaccf702, created=1536654688106, modified=1536654688106, origin=0]]], resources=[org.edgexfoundry.domain.meta.ProfileResource@35c30cc9, org.edgexfoundry.domain.meta.ProfileResource@6b57b74e, org.edgexfoundry.domain.meta.ProfileResource@16f5904b]]]!!!!motorpower
