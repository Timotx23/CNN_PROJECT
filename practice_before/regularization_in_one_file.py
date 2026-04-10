# Import necesary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from torch.utils.data import DataLoader



# Set random seed for reproducibility
torch.manual_seed(42)
def to_devices():

    # Check if GPU is available and set device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
  
    return device
  
#Actual CNN work
# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels= 3 , out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) #adding batch norm
        
        # Second Convolutional Layer
        # remember to calculate the in_channels based on the out_channels of the previous layer
        self.conv2 = nn.Conv2d(in_channels= 16 , out_channels= 32, kernel_size=3, padding=1) ## YOUR CODE HERE ##
        self.bn2 = nn.BatchNorm2d(32) #adding batch norm
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) #adding batch norm

        # Activation and Pooling
        self.relu = nn.ReLU()## YOUR CODE HERE ##
        self.pool = nn.MaxPool2d(kernel_size = 2, stride= 2) ## YOUR CODE HERE ## ( kernel_size=2, stride=2 )
        
        # Fully Connected Layers
        # calculate the in_features based on the FLATTENED output size after convolutions and pooling
        # calculate out_features based on the input size of the second fully connected layer
        fc1_in_features = 64 * 8 * 8 ## YOUR CODE HERE ##
        fc1_out_features =  64 ## YOUR CODE HERE ##
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=fc1_out_features)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second Convolutional Block
        x = self.conv2(x)

        x = self.relu(x)
        x = self.pool(x)
        
        # Third Convolutional Block
        x = self.conv3(x)
       
        x = self.relu(x)
        
        # Flatten for Fully Connected Layers
        x = x.view(-1, self.fc1.in_features)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


# Define a simple CNN model
class SimpleCNN_dropout(SimpleCNN):
    # use inheritance to avoid code duplication
    def __init__(self, dropout_prob=0.5):
        ## YOUR CODE HERE ##
        # dropout applied after the each convolutional layer
        super().__init__()
        
        self.dropout = nn.Dropout2d(p = dropout_prob) ## YOUR CODE HERE ##
        self.dropout_fc = nn.Dropout(p=dropout_prob)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)## YOUR CODE HERE ##
        x = self.pool(x)
        # Dropout applied after the first convolutional layer


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x) ## YOUR CODE HERE ##
        x = self.pool(x)
        # dropout applied after the second convolutional layer
        

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # dropout applied after the third convolutional layer
        x = self.dropout(x) ## YOUR CODE HERE ##
        
        # Flatten for Fully Connected Layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


