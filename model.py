
  import torch
  import torch.nn as nn
  import torchvision
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  class EmotionRecognizer(nn.Module):
def __init__(self, image_size, num_emotions):
  
  super(EmotionRecognizer, self).__init__()
  self.conv1 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
  )
  self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
  )
  self.drop_out = nn.Dropout()
  self.fc1 = nn.Linear(image_size[0] * image_size[1] * 64, 4500)
  self.fc2 = nn.Linear(4500, 1000)
  self.fc3 = nn.Linear(1000, num_emotions)
def forward(self, x):
  x = self.conv1(x)
  x = self.conv2(x)
  x = self.drop_out(x)
  x = self.fc1(x)
  x = self.fc2(x)
  x = self.fc3(x)

  er = EmotionRecognizer((48, 48), 10)
  er.to(device)
  
   