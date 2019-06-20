import os
import argparse
import cv2
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
ap.add_argument('--images', required=True, help='path to directory with images')
ap.add_argument('--method', required=False, help='averaging method, can be "Hard, Soft, Deep. default is Soft')
ap.add_argument('--threshold', required=False, help='threshold number or face recognition, default is 0.5')
args = vars(ap.parse_args())


if args['threshold'] is None or float(args['threshold']) >= 1 or float(args['threshold']) <= 0:
    threshold = 0.5
else:
    threshold = float(args['threshold'])

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

emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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
        if confidence < threshold:
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
        # get first place emotion
        max1 = out.max(0)[1]
        # prepare emotion string for printing
        emotion = emotions[max1]
        cv2.putText(frame, emotion, (startX-5, startY-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255))
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    if args['method'] is None:
        if args['method'] == 'Hard':
            avg_func = average.hard_average
        elif args['method'] == 'Soft':
            avg_func = average.soft_average
        elif args['method'] == 'Deep':
            avg_func = average.depth_sqrt_average
    else:
        avg_func = average.soft_average
    # assign the average function
    avg_func = average.depth_sqrt_average
    # evaluate the average using the given method
    avg = avg_func(predictions)
    max_average = avg.max(0)[1]
    emotion = emotions[max_average]
    cv2.putText(frame, emotion, (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
    new_image_path = os.path.join(save_path, image)
    cv2.imwrite(new_image_path, frame)
    