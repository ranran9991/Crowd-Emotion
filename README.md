# Crowd-Emotion

Dataset generation:

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

<<<<<<< HEAD
python cutface.py -p path/to/dataset/folder
=======
    python cutface.py

>>>>>>> cadd92cdf63b1fcd9ff81f3ac82a23e351350806

Note: I got preety poor results using this method, opencv did not manage to cut the faces very well and i got lots of false positives, it will probably be better to use existing datasets if possible
