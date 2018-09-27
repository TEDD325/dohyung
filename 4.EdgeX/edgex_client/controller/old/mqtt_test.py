import paho.mqtt.client as mqtt

MQTT_SERVER = 'localhost'

MQTT_STATE = 'state_info_from_edgex'

def __on_connect(client, userdata, flags, rc):
    print("mqtt broker connected with result code " + str(rc))
    client.subscribe(topic=MQTT_STATE)


def __on_message(client, userdata, msg):
    print(msg.topic + ' ' + str(msg.payload.decode("utf-8")))

sub = mqtt.Client(client_id="env_sub", transport="TCP")
sub.on_connect = __on_connect
sub.on_message = __on_message
sub.username_pw_set(username="link", password="0123")
sub.connect(MQTT_SERVER, 1883, 60)

print("Sub thread started!")
sub.loop_forever()
