import cv2
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import average
from prediction import Prediction

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((129.186279296875/255, 104.76238250732422/255, 93.59396362304688/255),  (1/255, 1/255, 1/255)),
    ]
)
ap = argparse.ArgumentParser()
ap.add_argument('--emotion', required=True, help='path to emotion net weights')
ap.add_argument('--face', required=True, help='path to face recognition net weights directory')
ap.add_argument('--images', required=True, help='path to directory with images')
args = vars(ap.parse_args())

class EmotionRecognizer(nn.Module):

    def __init__(self):
        super(EmotionRecognizer, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn55 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=4096, out_features=7, bias=True)

    def forward(self, data):
        x1 = self.conv1(data)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)
        x18 = self.pool5(x17)
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)
        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24_preflatten = self.relu7(x23)
        x24 = x24_preflatten.view(x24_preflatten.size(0), -1)
        prediction = self.fc8(x24)
        return prediction



net_path = args['face']
emotion_net_path = args['emotion']
images_path = args['images']

save_path = os.path.join(images_path, 'pred')
os.mkdir(save_path)

deploy = os.path.join(net_path, 'deploy.prototxt.txt')
params = os.path.join(net_path, 'res10_300x300_ssd_iter_140000.caffemodel')
# OpenCV model
net = cv2.dnn.readNetFromCaffe(deploy, params)

# load emotion model
emotion_net = EmotionRecognizer()
state_dict = torch.load(emotion_net_path, map_location='cpu')
emotion_net.load_state_dict(state_dict)

# evaluation mode
emotion_net.eval()

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

for image in os.listdir(images_path):
    print('Working on image: {}'.format(image))
    image_path = os.path.join(images_path, image)
    frame = cv2.imread(image_path)
    if frame is None:
        print('Skipped')
        continue

    (h, w) = frame.shape[:2]

    # transform image to blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    
    # detect faces in image
    net.setInput(blob)
    detections = net.forward()
    
    # faces will hold (X,Y) of top left and bottom right edges of bounding boxes in faces
    faces = []

    # iterate over detections to insert faces into faces list
    for i in range(0, detections.shape[2]):
        # confidence in detections
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        # threshhold can be changed
        if confidence < 0.80:
            continue

        # bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        faces.append((startX, startY, endX, endY))
    
    # iterate over captured faces
    predictions = []
    for (startX, startY, endX, endY) in faces:

        # crop image from original picture
        faceimg = frame[startY:endY, startX:endX].copy()
        try:
            # try to turn face into gray scale
            faceimg = cv2.cvtColor(faceimg, cv2.COLOR_BGR2GRAY)
        except:
            continue

        # prepare image for emotion recognizer
        faceimg = faceimg[:, :, np.newaxis]
        # make it have 3 identical channels
        faceimg = np.concatenate((faceimg, faceimg, faceimg), axis=2).astype(np.uint8)
        faceimg = Image.fromarray(faceimg)
        # apply transform
        faceimg = transform(faceimg)
        faceimg = faceimg[None]
        # get emotion confidence
        out = emotion_net.forward(faceimg)[0]
        # turn to probabilities
        out = nn.functional.softmax(out, dim=0)
        # create new prediction instance
        pred = Prediction(torch.Tensor(out), Prediction.BoundingBox(startX, startY, endX, endY))
        # insert the new prediction to the predictions list
        predictions.append(pred)
        # get first and second place emotions
        max1 = out.max(0)[1]
        out[max1] = 0.0
        max2 = out.max(0)[1]
        # prepare emotion string for printing
        emotion = '1.' + emotions[max1] + ' ' + '2.' + emotions[max2]
        cv2.putText(frame, emotion, (startX-5, startY-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255))
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # assign the average function
    avg_func = average.soft_average
    # evaluate the average using the given method
    avg = avg_func(predictions)
    max_average = avg.max(0)[1]
    emotion = emotions[max_average]
    cv2.putText(frame, emotion, (10, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255))
    new_image_path = os.path.join(save_path, image)
    cv2.imwrite(new_image_path, frame)
    