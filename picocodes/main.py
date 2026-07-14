import sys
import uselect
import time
import machine
from machine import SoftI2C, Pin

i2c0 = machine.I2C(0, sda=machine.Pin(0), scl=machine.Pin(1), freq=400000)
i2c1 = machine.I2C(1, sda=machine.Pin(2), scl=machine.Pin(3), freq=400000)
i2c2 = machine.SoftI2C(sda=machine.Pin(4), scl=machine.Pin(5), freq=400000)


DAC_ADDR = 0x60
led = Pin("LED", Pin.OUT)
devices0 = i2c0.scan()
devices1 = i2c1.scan()
devices2 = i2c2.scan()

if DAC_ADDR in devices0 and DAC_ADDR in devices1 and DAC_ADDR in devices2:
    print("DACs detected successfully!")
    for _ in range(3):
        led.value(1)
        time.sleep_ms(100)
        led.value(0)
        time.sleep_ms(100)
else:
    if DAC_ADDR not in devices0:
        print("DAC 1 not found")
    if DAC_ADDR not in devices1:
        print("DAC 2 not found")
    if DAC_ADDR not in devices2: 
        print("DAC 3 not found")


def set_dac_voltage(i2c_bus, digital_value):
    digital_value = max(0, min(digital_value, 4095))
    buffer = bytearray(3)
    buffer[0] = 0x40 
    buffer[1] = (digital_value >> 4) & 0xFF
    buffer[2] = (digital_value << 4) & 0xFF
    try:
        i2c_bus.writeto(DAC_ADDR, buffer)
    except:
        pass

upNdown = 1861
forwardNbackward = 1724
leftNright = 1724

set_dac_voltage(i2c0, upNdown)
set_dac_voltage(i2c1, forwardNbackward)
set_dac_voltage(i2c2, leftNright)

spoll = uselect.poll()
spoll.register(sys.stdin, uselect.POLLIN)

while True:
    if spoll.poll(0):
        line = sys.stdin.readline().strip()
        
        if len(line) > 1:
            prefix = line[0] 
            try:
                target_value = int(line[1:])
                
                if prefix == 'A':
                    set_dac_voltage(i2c0, target_value)
                elif prefix == 'B':
                    set_dac_voltage(i2c1, target_value)
                elif prefix == 'C':
                    set_dac_voltage(i2c2, target_value)
                    
            except ValueError:
                pass # Ignore malformed strings
                
    time.sleep(0.001)