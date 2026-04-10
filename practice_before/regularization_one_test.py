import regularization_in_one_file

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
import train_and_viz_cnn
train_loader, val_loader, test_loader, class_names = train_and_viz_cnn.get_data_loaders(None)
import regularization_in_one_file

#Prep work.
device = regularization_in_one_file.to_devices()

def visualize_data():
    train_and_viz_cnn.visualize_data(class_names, train_loader)

    # Extract labels from the subset
    subset_labels = []

    for _, labels in train_loader:
        # Assuming labels are a tensor, we convert them to a list and extend our accumulating list
        subset_labels.extend(labels.tolist())

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(subset_labels, bins=range(11), edgecolor='black', align='left', rwidth=0.8)
    plt.title('Histogram of Labels in the train set')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.xticks(range(10))
    plt.show()

    
def test_simple_cnn():
    # Simple test script for SimpleCNN initialization and forward pass
    # Create model instance
    model = regularization_in_one_file.SimpleCNN()

    # Create dummy input (batch_size=2, channels=3, height=32, width=32)
    dummy_input = torch.randn(2, 3, 32, 32)

    # Move model and input to the appropriate device
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Forward pass
    output = model(dummy_input)

    # Verify output shape
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (2, 10)")
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    print("✓ Model initialization and forward pass successful!")




def train_simple():

    # Setting Hyperparameters and Training the Model

    # Number of training epochs
    epochs = 40

    # Create an instance of the SimpleCNN model and move it to the specified device (GPU if available)
    model = regularization_in_one_file.SimpleCNN().to(device)
    print("Device", device)
    # Define the loss criterion (CrossEntropyLoss) and the optimizer (Adam) for training the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model using the defined training function
    val_losses_simple = train_and_viz_cnn.train_model(model, train_loader, val_loader, epochs, criterion, optimizer)
    return val_losses_simple


def test_dropout():
        # Simple test for SimpleCNN_dropout
    model_d = regularization_in_one_file.SimpleCNN_dropout(dropout_prob=0.5)
    print(f"Model created with dropout_prob={model_d.dropout_prob}")

    # Test with dummy input
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model_d(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 10)")
    assert output.shape == (2, 10), f"Shape mismatch: got {output.shape}"
    print("✓ SimpleCNN_dropout test passed!")

def train_with_dropout():
    epochs = 100
    model_d = regularization_in_one_file.SimpleCNN_dropout(dropout_prob=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_d.parameters(), lr=0.001)
    val_losses_dropout = train_and_viz_cnn.train_model(model_d, train_loader, val_loader, epochs, criterion, optimizer)
    return val_losses_dropout

def comparing_models():
    #Call models
    val_losses_simple = train_simple()
    val_losses_dropout = train_with_dropout()
    #val_losses_data_aug = train_with_data_augmentation()


    plt.plot(val_losses_simple, label='No regularization')
    plt.plot(val_losses_dropout, label='Dropout')
    #plt.plot(val_losses_data_aug, label='Data augmentation + Dropout')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()







def train_with_data_augmentation(model_d1, epochs):
    # Set the number of epochs and batch size

    # Create a CNN model with dropout
    #model_d1 = regularization_in_one_file.SimpleCNN_dropout(dropout_prob=0.3).to(regularization_in_one_file.to_device()) # Changed dropout probability to 0.3 since 0.5 was too high in combination with data augmentation
   
    print("Device", device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1 )
    optimizer2 = optim.Adam(model_d1.parameters(), lr=0.001)# old
    optimizer = torch.optim.Adam(model_d1.parameters(), lr=0.001, weight_decay=1e-4) #normal adam has given the best results
    #optimizer 3
    optimizer3 = torch.optim.AdamW(model_d1.parameters(), lr=0.001, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)


    # Define data augmentation transformations for training (hint: think about what transformations can help with learning more robust reopresentations)
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
]) #YOUR CODE HERE #

    # Create data loaders for training, validation, and testing using the defined transformations in transform_train
    train_loader, val_loader, test_loader, _ = train_and_viz_cnn.get_half_db(transform_train, batch_size=128)# YOUR CODE HERE #
    
    val_losses_data_aug = train_and_viz_cnn.train_model(model_d1, train_loader, val_loader, epochs, criterion, optimizer, scheduler)
    return val_losses_data_aug, test_loader






# Test the Model
def test_model(model, test_loader):

    # Set the model to evaluation mode
    model.eval()
    
    # Initialize counters
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # Disable gradient computation during testing
    with torch.no_grad():
        # Iterate through the test loader
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for further analysis
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    # Calculate accuracy
    accuracy = correct / total

    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='macro')

    # Build confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix with class names
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Display test results
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f"F1 = {f1:.2f}")

    return accuracy, f1



model_d1 = regularization_in_one_file.SimpleCNN_dropout(dropout_prob=0.2).to(device)
epochs = 200
val_looses_data_aug, test_loader = train_with_data_augmentation(model_d1, epochs)
acc, f1 = test_model(model_d1, test_loader)
