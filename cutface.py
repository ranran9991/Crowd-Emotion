import argparse
import os
import cv2

def cut_faces(path):
    """
        Recursively cuts faces off of images in path
        Note that this function overwrites existing images
    """
    for file_name in os.listdir(path):
        # if file is folder, recursivly cut faces
        if os.path.isdir(os.path.join(path, file_name)):
            cut_faces(os.path.join(path, file_name))        
        f = os.path.join(path, file_name)
        f_name = os.path.basename(f)

        # if file is image
        img = cv2.imread(f)
        if img is None:
            continue
        print('[INFO] working on ' +  f)

        # capture faces in image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier("C:\\Users\\ranran9991\\Envs\\env\\haarcascade_frontalface_default.xml")
        faces, num_detections = face_cascade.detectMultiScale(img, 1.1, 5)
        if num_detections == 0:
            os.remove(f)
            continue
        # iterate over faces
        i = 0
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h /2
            nx = int(centerx - r)
            ny = int(centery - y)
            nr = int(r * 2)
            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (64, 64))
            if num_detections == 1:
                cv2.imwrite(os.path.join(path, f_name), lastimg)
            else:
                cv2.imwrite(os.path.join(path, str(i), f_name), lastimg)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	            help="path to output directory of images")

args = vars(ap.parse_args())
dataset_dir = args['path']

cut_faces(dataset_dir)

