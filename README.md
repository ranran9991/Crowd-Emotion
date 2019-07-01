# Crowd-Emotion
## Datasets:
1. FER2013
 - Download the dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
2. CK+
 - Download the dataset from [here](http://www.consortium.ri.cmu.edu/ckagree/)
 - Run the script cutface.py with the -p / --path argument 
 - python cutface.py -p path/to/dataset/folder
3. Generate the dataset yourself

### Dataset generation:

Requirements:
 - python 3.6 
 - requests (downloaded with pip, "pip install requests")
 - OpenCV (downloaded with pip, latest version)

The dataset is generated using Microsofts Cognitive Services API, specifically, Bing Image Search API, so you should get an API key from [here](https://azure.microsoft.com/en-us/try/cognitive-services/my-apis/?api=bing-image-search-api)

Run the python script dataset_prep.py with the following flags arguments:

 -o / --output path/to/desired/output/directory
 
 -k / --key API key 

    python dataset_prep.py -o path/to/desired/output/directory -k YOUR_KEY_HERE

This script should download the photos into the output directory

Next, you should look over the folders and remove pictures that are not suitble, even though the next script should probably take care of such pictures.

run the cutface.py script with the -p / --path argument. This would grayscale the images and cut the faces from the pictures, removing the old pictures in the process.

    python cutface.py -p path/to/dataset/folder

Note: I got preety poor results using this method, opencv did not manage to cut the faces very well and i got lots of false positives, it will probably be better to use existing datasets if possible


## Dataset usage
you should use the dataset using the `EmotionDataset` class found in emotion_dataset_class.py and in conjunction with pytorch's DataLoader class like so:
```python
# define transforms over dataset
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
# For FER2013 this should be the path to the csv file
# For CK+ this shold be the path to the directory containing the 'cohn-kanade-images' and 'Emotion' directories
# For the generated dataset this should be the path to the directory containing a directory for each emotion
path = 'path\\to\\dataset'
# can either be 'FER2013', 'CK+'. anything else use take the generated dataset
dataset_type = 'CK+'
# can either be 'Training' or 'Validation'
split = 'Training'

# define dataset
dataset = EmotionDataset(path,
                         dataset_type=dataset_type,
                         split=split,
                         transform=transform)
                         
# create data loader
training_loader = DataLoader(dataset, batch_size=1, num_workers=4)
```

## Running the script
### Requirements
 - Python 3.6
 - OpenCV (downloaded with pip)
 - Pytorch (1.1 was used, should work with newer versions)
 
 All the scripts require two things:
 - Weights of the face detection NN (in the utils folder)
 - Weights of the emotion recognition NN (download from [here](http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_m_face_bn_fer_dag.pth)
 
 ### Recognizing emotions from files
 Getting predictions on a folder filled with images
 
 Run the following command
 
      python recognize_files.py --emotion /utils --face path/to/emotion/net/weights --images path/to/image/folder
      
 You could also set the averaging method by adding the flag --method Hard/Soft/Deep. default is Soft
 
 You could also set the confidence threshold for the face recognition dnn by adding the flag --threshold, default is 0.5
 ### Recognizing emotions from camera
 Getting predictions from the camera
 
 Run the following command 
 
     python recognize_from_camera.py --emotion /utils --face path/to/emotion/net/weights
 It uses the first available camera, on a laptop that will be the laptop builtin camera, To other cameras add the flag --camera <camera_num>, 0 for first camera, then 1,2,3...
 
 ## Training
 The training uses the FER2013 dataset, it assumes you the fer2013.csv file in the current working directory
 Run
 
     python train.py 
 The script saves a file called `fer2013_model.pt` which holds the weights of the trained model.
 
