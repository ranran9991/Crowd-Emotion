import sys
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import average
from prediction import Prediction
from model import EmotionRecognizer

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
ap.add_argument('--camera_num', required=False, help='Camera number, 0 for laptop webcam, 1,2,... for connected camera, Default is 0')
args = vars(ap.parse_args())

# enter the full path to a classifier xml file, for example:
# face recognition model
net_path = args['face']
# emotion classification model
emotion_net_path = args['emotion']

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

if args['camera_num'] != None:
    video_capture = cv2.VideoCapture(int(args['camera_num']))
else:
    # first available camera, for laptops this will be the laptop cam
    video_capture = cv2.VideoCapture(0)

while True:
    # capture an image
    ret, frame = video_capture.read()
    cv2.imshow('Crowd Emotion', frame)
    k = cv2.waitKey(1)
    # if q was clicked
    if k & 0xFF == ord('q'):
        exit()
    # if y was clicked
    if k & 0xFF == ord('y'):
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
            out_temp = out.clone().detach()
            max1 = out_temp.max(0)[1]
            out_temp[max1] = 0.0
            max2 = out_temp.max(0)[1]
            # prepare emotion string for printing
            emotion = '1. ' + emotions[max1] + ' ' + '2.' + emotions[max2]
            cv2.putText(frame, emotion, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # assign the average function
        avg_func = average.soft_average
        # evaluate the average using the given method
        avg = avg_func(predictions)
        max_average = avg.max(0)[1]
        emotion = emotions[max_average]
        cv2.putText(frame, emotion, (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
        # show image
        cv2.imshow('Pred', frame)
        k = cv2.waitKey(0)
        # press q to close the window
        if k & 0xFF == ord('q'):
            exit()
        if k & 0xFF == ord('a'):
            cv2.destroyWindow('Pred')
            continue

video_capture.release()
cv2.destroyAllWindows()
