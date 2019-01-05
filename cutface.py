import argparse
import os
import numpy as np
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	            help="path to output directory of images")

args = vars(ap.parse_args())
dataset_dir = args['path']

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('utils\\deploy.prototxt.txt', 'utils\\res10_300x300_ssd_iter_140000.caffemodel')




def cut_faces(path):
    """
        Recursively cuts faces off of images in path
        Note that this function overwrites existing images
    """
    for file_name in os.listdir(path):
        # if file is folder, recursivly cut faces
        if os.path.isdir(os.path.join(path, file_name)):
            cut_faces(os.path.join(path, file_name))     
            continue
     
        f = os.path.join(path, file_name)
        f_name = os.path.basename(f)

        # read image
        img = cv2.imread(f)
        # if image can't be opened by opencv
        if img is None:
            os.remove(f)
            continue
            
        print('[INFO] working on ' +  f)
        (h, w) = img.shape[:2]
        # transform image to blob
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	        (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()
        faces = []
        # iterate over detections to insert faces into faces list
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence < 0.65:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
            # faces.append((startX, startY, w, h))

        num_detections = len(faces)
        print('\t' + str(num_detections) + ' detected in picture')
        if num_detections == 0:
            os.remove(f)
            continue

        # remove the original picture 
        os.remove(f)
        # iterate over faces
        for i, (startX, startY, endX, endY) in enumerate(faces):
            # crop image
            faceimg = img[startY:endY, startX:endX].copy()
            # resize to (64,64)
            lastimg = cv2.resize(faceimg, (64, 64))
            # turn to grayscale
            lastimg = cv2.cvtColor(lastimg, cv2.COLOR_BGR2GRAY)
            #save
            if num_detections == 1:
                cv2.imwrite(os.path.join(path, f_name), lastimg)
            else:
                crop_name = str(i) + '_' + f_name
                cv2.imwrite(os.path.join(path, crop_name), lastimg)
            
            

cut_faces(dataset_dir)
cv2.waitKey(0)