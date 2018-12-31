from torch.utils.data import DataLoader
import torchvision
from emotion_dataset_class import EmotionDataset

if __name__ == "__main__":
    # define transforms over dataset
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    path = 'C:\\Users\\ranran9991\\Desktop\\Crowd-Emotion\\dataset\\CK+'
    # can either be 'FER2013', 'CK+'. anything else will take the generated dataset
    dataset_type = 'CK+'
    # can either be 'Training' or 'Validation'
    split = 'Training'
    
    # define dataset
    dataset = EmotionDataset(path,
                             dataset_type='CK+',
                             split='Training',
                             transform=transform)
    print('Number of examples in dataset is: ' + len(dataset))
    # create data loader
    training_loader = DataLoader(dataset, batch_size=1, num_workers=4)
    """
    # iterate over dataset
    for index, (images, labels) in enumerate(training_loader):
            print(images, labels)
     
    """
           


