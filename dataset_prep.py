import argparse
import os
import sys
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
ap.add_argument("-k", "--key", required=True,
help="API key for the Microsoft Cognitive Services API")

args = vars(ap.parse_args())

emotions = ['happy', 'angry', 'neutral', 'sad', 'afraid', 'disgusted']
queries = ["\"" + emotion + " person face\"" for emotion in emotions]

path = args['output']
paths = []
for emotion, query in zip(emotions,queries):
    emotion_path = os.path.join(path, emotion)
    paths.append(emotion_path)
    if not os.path.exists(emotion_path):
        os.makedirs(emotion_path)
    os.system('python download_dataset.py --query ' + query + ' --output ' + emotion_path + ' --key ' + args['key'])
