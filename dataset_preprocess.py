import csv
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def preprocess_generated(path):
    """
        Takes path to generated dataset and returns list of labeld training and validation data
    """
    data_x = []
    data_y = []
    # iterating over emotions
    for index, emotion_name in enumerate(os.listdir(path)):
        emotion_path = os.path.join(path, emotion_name)
        # iterating over images in emotion
        for image_name in os.listdir(emotion_path):
            image_pixels = (np.array(Image.open(os.path.join(emotion_path, image_name)), dtype=np.int)).flatten()
            image_shape = int(np.sqrt(len(image_pixels)))
            image_pixels = np.reshape(image_pixels, (image_shape, image_shape))
            data_x.append(image_pixels)
            data_y.append(index)
    # splitting the data to training and validation

    training_x, validation_x, training_y, validation_y = train_test_split(data_x, data_y,
                                                                          test_size=0.2,
                                                                          shuffle=True)

    return training_x, training_y, validation_x, validation_y

def preprocess_fer2013(path):
    """
        Takes fer2013.csv file path and returns list of labeld training and validation data
    """
    training_x = []
    training_y = []
    validation_x = []
    validation_y = []

    with open(path, 'r') as csvin:
        data = csv.reader(csvin)
        for row in data:
            # if we read the first line which holds no valuable data
            if(row[1] == 'pixels'):
                continue
            # reads image bytes
            image_bytes = (np.asarray([int(x) for x in (row[1].split())])).reshape((48, 48))
            if row[-1] == 'Training':
                training_x.append(image_bytes)
                training_y.append(int(row[0]))
            if row[-1] == 'PublicTest' or row[-1] == 'PrivateText':
                validation_x.append(image_bytes)
                validation_y.append(int(row[0]))
    
    return training_x, training_y, validation_x, validation_y

def preprocess_ckp(path):
    """
        takes path to the home path to the CK+ dataset and returns list of labeled training and validation data
    """
    data_x = []
    data_y = []
    files = os.listdir(path)
    files.sort()
    assert files[0] == 'Emotion'
    assert files[1] == 'cohn-kanade-images'
    # grabs images and labels
    data_x, data_y = grab_images_and_lables(os.path.join(path, files[1]), os.path.join(path, files[0]))
    # reshapes them 
    shape = int(np.sqrt(len(data_x[0])))
    data_x = [np.reshape(x, (shape, shape)) for x in data_x]
    # split to training and validation
    training_x,validation_x, training_y, validation_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True)
    
    return training_x, training_y, validation_x, validation_y

    
def grab_images_and_lables(path_to_images, path_to_labels):
    """
        helper recursive function for preprocess_ckp
    """
    data = []
    labels = []

    files = (os.listdir(path_to_images))
    files.sort()
    for f in files:
        if f[0] == '.':
            continue
        # if file is directory
        if os.path.isdir(os.path.join(path_to_images,f)):
            subdir_data, subdir_lables = grab_images_and_lables(os.path.join(path_to_images, f), os.path.join(path_to_labels, f)) 
            data += subdir_data
            labels += subdir_lables
        # list of images
        else:
            
            # takes the middle image
            image_index = int(len(files) / 2)
            image = os.path.join(path_to_images, files[image_index])
            image_pixels = (np.array(Image.open(image))).flatten()
            try:
                # get label from text file
                label_file = os.path.join(path_to_labels, os.listdir(path_to_labels)[0]) 
                txtfile = open(label_file, 'rb')
                lines = txtfile.read().strip()
                txtfile.close()
                label = int(float(lines))
                return [image_pixels], [label]
            except:
                continue
    return data, labels
            
