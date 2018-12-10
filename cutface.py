import argparse
import os
import sys
import cv2

paths = ['C:\\Users\\ranran9991\\Desktop\\dataset\\happy']



for class_path in paths:
    for image in os.listdir(class_path):
        print('[INFO] working on' +  image)
        image = os.path.join(class_path, image)
        name = os.path.basename(image)

        img = cv2.imread(image)
        if (img is None):
            os.remove(image)
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier("C:\\Users\\ranran9991\\Envs\\env\\haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(img, 1.2, 5, minSize=(100,100))
        if(faces is None):
            os.remove(image)
            continue
        facecnt = len(faces)
        i = 0
        height, width = img.shape[:2]
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h /2
            nx = int(centerx - r)
            ny = int(centery - y)
            nr = int(r * 2)
            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (64,64))
            i +=1
            cv2.imwrite(os.path.join(class_path ,name + "%d.jpg" %i), lastimg)
        os.remove(image)