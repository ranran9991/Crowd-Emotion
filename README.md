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
# For the generated dataset this should be the path to the directory containing a directory for each image
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
