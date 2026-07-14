import numpy as np
import torch
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F 
from model import CNN
from PIL import Image
import serial

# ----------------------------------- CNN -----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 20
learning_rate = 0.0005
batch_size = 32

MODEL_SAVE_PATH = r"C:\Users\Andre\OneDrive\Desktop\codes\drone_project\gesture_cnn.pth"
FORCE_RETRAIN = False 

cnn = CNN(num_classes=7).to(device)


def train_model():
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    class AddGaussianNoise:
        def __init__(self, mean=0., std=0.05):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        AddGaussianNoise(mean=0., std=0.05)  # add after ToTensor since it operates on tensors
    ])

    training_data_path = r"C:\Users\Andre\OneDrive\Desktop\codes\drone_project\training_data"
    training_data = datasets.ImageFolder(root=training_data_path, transform=data_transforms)
    train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    print("Folder to Label Mapping:", training_data.class_to_idx)

    cnn.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # forward pass: feed images into the CNN
            outputs = cnn(images)
            loss = loss_fn(outputs, labels)

            # Backward pass: optimize the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_epoch_loss:.4f}")

    # save the trained weights so we never have to do this again
    torch.save(cnn.state_dict(), MODEL_SAVE_PATH)
    print(f"Training finished successfully — model saved to {MODEL_SAVE_PATH}")


# Load the saved model if it exists, otherwise train once and save it
if os.path.exists(MODEL_SAVE_PATH) and not FORCE_RETRAIN:
    cnn.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"[SUCCESS] Loaded trained model from {MODEL_SAVE_PATH} — skipping training")
else:
    print("[INFO] No saved model found (or FORCE_RETRAIN=True) — training now...")
    train_model()


# ------------------------------------------------ CONNECT TO PICO ------------------------------------------------------------

try:
    s = serial.Serial('COM3', 115200, timeout=0.1)
    print("[SUCCESS] Connected to Pico successfully!")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    exit()


def send_command(prefix, val):
    s.write(f"{prefix}{val}\n".encode())


# start with every axis centered, like a joystick at rest
send_command('A', 1861)
send_command('B', 1724)
send_command('C', 1724)


# ------------------- PRESS / RELEASE SIMULATION (the keyboard mechanism, but for gestures) -------------------


GESTURE_PRESS = {
    'up':         ('A', 1400),  
    'down':       ('A', 2400),  
    'forward':    ('B', 1350),
    'backward':   ('B', 2150),
    'turn-left':  ('C', 1350),   
    'turn-right': ('C', 2150),   
}
AXIS_CENTER = {'A': 1861, 'B': 1724, 'C': 1724}
CONFIRM_FRAMES = 5 # prediction must repeat for COMFIRM_FRAMES number of frames in order to count it as a move
CONFIDENCE_MIN = 0.75        # ignore predictions below this confidence
NO_HAND_RELEASE_FRAMES = 10  # hand off-screen for this many frames = release everything

active_gesture = 'hover'     # the gesture currently being "held"
candidate = None             # gesture we're considering switching to
candidate_count = 0          # how many consecutive frames we've seen it
no_hand_count = 0 # how many consecutive frames without hands

# gesture becomes active
def press(gesture):
    if gesture in GESTURE_PRESS:
        axis, val = GESTURE_PRESS[gesture]
        # print(f"[PRESS]   {gesture} -> {axis}{val}")
        send_command(axis, val)

# deactivate gesture
def release(gesture):
    if gesture in GESTURE_PRESS:
        axis, _ = GESTURE_PRESS[gesture]
        # print(f"[RELEASE] {gesture} -> {axis}{AXIS_CENTER[axis]} (center)")
        send_command(axis, AXIS_CENTER[axis])


def set_gesture(new_gesture):
    global active_gesture
    if new_gesture == active_gesture:
        return  # same key still held -> send nothing
    # deactivate old gesture
    release(active_gesture)
    # activate new gesture
    press(new_gesture)
    active_gesture = new_gesture


# -------------------------------------------------- MEDIAPIPE + WEBCAM SETUP ------------------------------------------------------------

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((100, 100)),   
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

inf = 100000000000
model_path = r"C:\Users\Andre\OneDrive\Desktop\codes\drone_project\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)

detertor = HandLandmarker.create_from_options(options)
wemcam = cv2.VideoCapture(0)

# crops hand from the webcam footage
def crop_hand(frame, mnx, mxx, mny, mxy, target_size=400):
    h, w, _ = frame.shape
    half_size = target_size // 2

    # calculate the center point of the detected hand landmarks
    center_x = (mnx + mxx) // 2
    center_y = (mny + mxy) // 2

    # calculate crop coordinates around the center
    y1 = center_y - half_size
    y2 = center_y + half_size
    x1 = center_x - half_size
    x2 = center_x + half_size

    # shift the y-axis window if it hits the top or bottom border
    if y1 < 0:
        y2 = min(h, y2 + abs(y1))
        y1 = 0
    elif y2 > h:
        y1 = max(0, y1 - (y2 - h)) 
        y2 = h

    # shift the x-axis window if it hits the left or right border
    if x1 < 0:
        x2 = min(w, x2 + abs(x1))  
        x1 = 0
    elif x2 > w:
        x1 = max(0, x1 - (x2 - w)) 
        x2 = w

    hand_img = frame[y1:y2, x1:x2, :]
    return hand_img


gestures = ['backward', 'down', 'forward', 'hover', 'turn-left', 'turn-right', 'up']
cnn.eval()

# -------------------------------------------------- MAIN LOOP ------------------------------------------------------------

while True:
    ret, frame = wemcam.read()

    h, w, _ = frame.shape
    # so we must convert it before processing it with mediapipe
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    res = detertor.detect(mp_image)
    mnx, mxx, mny, mxy = w, 0, h, 0
    skeleton_frame = np.zeros_like(frame)
    # this the part where mediapipe crops the hand from frame
    if res.hand_landmarks and res.handedness:
        skeleton_frame = np.zeros_like(frame)
        for i in range(len(res.hand_landmarks)):
            handinfo = res.handedness[i][0]
            label = handinfo.category_name
            # confidence = handinfo.score
            tx = int(res.hand_landmarks[i][12].x * w)
            ty = inf
            for landmark in res.hand_landmarks[i]:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                if label=="Left":
                    mnx=min(mnx,x)
                    mxx=max(mxx,x)
                    mny=min(mny,y)
                    mxy=max(mxy,y)
                ty = min(ty, y)
                cv2.circle(skeleton_frame, (x, y), 5, (255, 0, 0), cv2.FILLED)

            for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                start_idx = connection.start
                end_idx = connection.end
                start = res.hand_landmarks[i][start_idx]
                end = res.hand_landmarks[i][end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(skeleton_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # if (ty-20 >= 0): cv2.putText(skeleton_frame, str(label), (tx,ty-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hand_cropped = crop_hand(skeleton_frame, mnx, mxx, mny, mxy, 300)

    if mnx < w:
        # hand detected this frame
        no_hand_count = 0

        tmp = cv2.cvtColor(hand_cropped, cv2.COLOR_BGR2RGB)  # BGR to RGB
        tmp = Image.fromarray(tmp)                           # numpy to PIL
        input_tensor = test_transforms(tmp).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = cnn(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            prediction_gesture = gestures[prediction.item()]
            print(prediction_gesture)
            # if confidence is greater then the min bound
            if confidence.item() >= CONFIDENCE_MIN:
                if prediction_gesture == candidate:
                    candidate_count += 1
                else:
                    candidate = prediction_gesture
                    candidate_count = 1
                if candidate_count >= CONFIRM_FRAMES:
                    set_gesture(candidate)
    else:
        # no hand on screen = hands off the keyboard -> release everything
        candidate = None
        candidate_count = 0
        no_hand_count += 1
        if no_hand_count >= NO_HAND_RELEASE_FRAMES:
            set_gesture('hover')

    cv2.imshow('frame', frame)
    if mnx < w: cv2.imshow('hand', hand_cropped)
    if cv2.waitKey(1) == ord('q'):
        print("Exiting... Resetting positions.")
        send_command('A', 1861)
        send_command('B', 1724)
        send_command('C', 1724)
        break

wemcam.release()
cv2.destroyAllWindows()
s.close()
print("Disconnected.")