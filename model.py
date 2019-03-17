import torch
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.optim as optim
import torchvision
from emotion_dataset_class import EmotionDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#HYPER PARAMETERS
LR = 0.0001
WEIGHT_DECAY = 0
EPOCHS = 200
BATCH_SIZE = 128
transform = {
    'train' : torchvision.transforms.Compose(
    [
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(40),
     torchvision.transforms.ToTensor(),
    ]
    ),
    'validation' : torchvision.transforms.Compose(
    [
     torchvision.transforms.ToTensor(),
    ]
    )
}

class EmotionRecognizer(nn.Module):
  def __init__(self, image_size, num_emotions):

    super(EmotionRecognizer, self).__init__()
    self.image_size = image_size
    
    self.conv = nn.Sequential(
      nn.BatchNorm2d(3),
      nn.Conv2d(3, 64, kernel_size=5),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
      
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(),
       
      nn.BatchNorm2d(128),  
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=3, stride=2),
      
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.ReLU(),
        
      nn.BatchNorm2d(128),  
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    )
    
    self.fc = nn.Sequential(
        nn.BatchNorm1d(3200),
        nn.Linear(3200, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        
        nn.BatchNorm1d(1024),
        nn.Linear(1024, num_emotions),
    )
    
    
  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
  
# create net instance
net = EmotionRecognizer((48, 48), 7)
net.to(device)

# create dataset and wrappers
path = 'fer2013.csv'

train_dataset = EmotionDataset(path, dataset_type='FER2013', split='Training', transform=transform['train'])
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=4)
val_dataset = EmotionDataset(path, dataset_type='FER2013', split = 'Validation', transform=transform['validation'])
val_loader = DataLoader(val_dataset, batch_size = 1)

# Loss function
# note that CrossEntropyLoss applies softmax and NLL loss
loss = torch.nn.CrossEntropyLoss()

#Optimizer
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay = WEIGHT_DECAY)
best_acc = 0
n_batches = len(train_loader)
#Loop for n_epochs
for epoch in range(EPOCHS):
  print("Epoch number {}".format(epoch+1))
  net = net.train(True)
  running_loss = 0.0
  print_every = n_batches/2
  start_time = time.time()
  total_train_loss = 0

  for i, data in enumerate(train_loader, 0):

    #Get inputs
    inputs, labels = data

    #Wrap them in a Variable object
    inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

    #Set the parameter gradients to zero
    optimizer.zero_grad()

    #Forward pass, backward pass, optimize
    outputs = net(inputs)
    loss_size = loss(outputs, labels)
    loss_size.backward()
    optimizer.step()

    #Print statistics
    running_loss += loss_size.data.item()
    total_train_loss += loss_size.data.item()

    if (i + 1) % (print_every + 1) == 0:
        print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
    #Reset running loss and time
    running_loss = 0.0
    start_time = time.time()

  #At the end of the epoch, do a pass on the validation set
  net.eval()
  net.train(False)
  total_val_loss = 0
  accuracy = 0
  pred_labels = []
  true_labels = []
  softmax = nn.Softmax(dim=0)
  for inputs, labels in val_loader:

    #Wrap tensors in Variables
    inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

    #Forward pass
    val_outputs = net(inputs)
    pred_label = torch.argmax(softmax(val_outputs[0])).item()
    pred_labels.append(pred_label)
    true_labels.append(labels.item())
    if pred_label == labels:
      accuracy = accuracy + 1
    val_loss_size = loss(val_outputs, labels)
    total_val_loss += val_loss_size.item()
  
  accuracy = accuracy * 100 / len(val_loader)
  print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
  print("Validation accuracy = {:.6f}%".format(accuracy))
  cm = confusion_matrix(true_labels, pred_labels)
  print(cm)
  if(accuracy > best_acc):
    best_acc = accuracy
    torch.save(net, 'fer2013_model.pt')
