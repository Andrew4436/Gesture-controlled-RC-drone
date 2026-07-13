# Gesture-controlled-RC-drone

This project lets me fly a drone with hand gestures through a webcam.

## Demo
https://github.com/user-attachments/assets/32db902f-7ea7-4360-876e-85e2518d3ba5


## Collecting the data

I collected 7 different hand gestures, one for each movement: forward, backward, up, down, left, right and hover. Each gesture has around 1000 images for the CNN to train on.

## Training the CNN

I used PyTorch to build a CNN and trained it on those images. I tried couple different ways to optimize my CNN.

First I tried cropping just the small area around the hand, so the background wouldn't affect the CNN, but there were still enough background for CNN to get distracted by.

Then I took the cropped image and grayscaled it, i thought that this would make it so that lighting or color wouldnt be a factor when training the CNN, but that wasnt the case.

Then I tried a skin filter, anything in the image that isn't skin tone automatically gets filtered out and changed to a completely black pixel. But I realized this relies heavily on the lighting, so it wasn't reliable either.

Then it realized that only thing that really matters is the shape of the hand, not anything else. So using the 21 hand positions from MediaPipe, for each cropped image I replaced the hand with a hand skeleton model. Here's a comparison of the three methods:

<img width="894" height="291" alt="Image" src="https://github.com/user-attachments/assets/e238eb74-1965-4a3e-ae2f-4d91ac697b85" />

*Left: greyscaled image. Center: skeleton model. Right: skin filter.*

In my experience the skeleton model worked by far the best, since it only preserves the shape of the hand. With this I was able to get 94% accuracy testing on live, unseen webcam footage.

## Controlling the drone

Once the CNN is trained, here's how it actually controls the drone.

Whenever the CNN recognizes a gesture, it sends a message to a Raspberry Pi Pico. The Pico receives a number from 0 to 4096, translate that number into voltage and sends it to a DAC (digital to analog converter). The voltage then sent to the drone controller, which is connected to the DAC through jumper wires. This basically tricks the controller into thinking someone actually pressed the joystick.

the formula to translate the number 0 to 4096 into the voltage is

Since there are 3 axes of movement (up/down, forward/backward, left/right), there are 3 DACs, one for each axis. Whenever a number is sent to the Pico, the Pico handles which DAC it goes to. For example, to move up, the message is `('A', 1300)`: `'A'` says which DAC to send the voltage to, and `1300` represents the amount of voltage, not translated yet.
