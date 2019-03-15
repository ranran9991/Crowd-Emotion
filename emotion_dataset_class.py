import numpy as np
import torch.utils.data as data
from PIL import Image
from dataset_preprocess import preprocess_ckp, preprocess_fer2013, preprocess_generated


class EmotionDataset(data.Dataset):
    """
        emotion dataset class, used in conjunction with the pytorch dataloader
    """

    def __init__(self, path, dataset_type='CK+', split='Training', transform=None):
        """
            dataset_type = either 'CK+', 'FER2013' or generated type
            path - path to the dataset
            split - Training / Validation
            transform - transformation over the dataset
        """
        self.transform = transform
        self.split = split
        if dataset_type == 'CK+':
            training_data, training_labels, validation_data, validation_labels = preprocess_ckp(path)
        elif dataset_type == 'FER2013':
            training_data, training_labels, validation_data, validation_labels = preprocess_fer2013(path)
        else:
            training_data, training_labels, validation_data, validation_labels = preprocess_generated(path)

        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

        if self.split == 'Training' and dataset_type == 'FER2013':

            self.training_data = (np.asarray(training_data)).reshape((28709, 48, 48))
            self.training_labels = training_labels
        
        if self.split == 'Validation' and dataset_type == 'FER2013':
            self.validation_data = (np.asarray(validation_data)).reshape((7178, 48, 48))
            self.validation_labels = validation_labels

    def __getitem__(self, index):
        """
            Gets element of data at index
            returns tuple (image, label)
        """
        if self.split == 'Training':
            img, target = self.training_data[index], self.training_labels[index]
        else:
            img, target = self.validation_data[index], self.validation_labels[index]
                 
        # doing this so that it is consistent with all other datasets
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        """
            returns length of dataset
        """
        if self.split == 'Training':
            return len(self.training_data)
        else:
            return len(self.validation_data)
