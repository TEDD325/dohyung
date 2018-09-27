import RPi.GPIO as gpio
import time

led_pin = 2
gpio.setmode(gpio.BCM)
gpio.setup(led_pin, gpio.OUT)

while True:
    gpio.output(led_pin, True)
    time.sleep(0.1)
    gpio.output(led_pin, False)
    time.sleep(0.1)

gpio.cleanup()