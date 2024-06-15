"""
This module contains the implementation of a training environment for
a neural network model using PyTorch.

It includes the definition of the neural network architecture (LeNet),
setting up the training context,
handling the training and validation processes, and saving the trained
model and its training history.

Classes:
    LeNet: Implements a simple convolutional neural network architecture.
    TrainingHistory: Manages and stores the training history.
    TrainingContext: Configures and holds the context for the training session.

Functions:
    train_network(initial_learning_rate, epochs, train_loader, validation_loader, classes):
    Manages the training process.
    save_model(model, path): Saves the trained model to a file.
    save_history_to_csv(history, path): Saves the training history to a CSV file.
    visualisation_of_history(history): Displays the training and validation accuracy.

How to use the module:
    - Configure the paths, and hyperparameters.
    - Instantiate the data loaders.
    - Call the `train_network` function with appropriate parameters.
"""
# pylint: disable=import-error
import argparse
import logging
import time
import csv
import os
from typing import cast, Sized
import torch
import matplotlib.pyplot as plt

from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, LogSoftmax
from torch.utils.data import DataLoader, random_split
from torch import flatten
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder


#==================================================================================================
#                                            LOGGER
#==================================================================================================

logger = logging.getLogger()
lprint = logger.info

def setup_logger(dataset_path: str|None = None)-> None:
    """Sets up the logger for logging information to console and file.

    Args:
        dataset_path (str | None, optional): Path to the dataset directory. Defaults to None.
    """
    log_formatter = logging.Formatter('%(message)s')

    if dataset_path:
        logfile_path = os.path.join(dataset_path, 'dataset.log')
        file_handler = logging.FileHandler(logfile_path)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)

def print_text_separator():
    """Prints a separator line to the log."""
    lprint('--------------------------------------------------------')

#==================================================================================================
#                                         NEURAL NETWORKS
#==================================================================================================
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class LeNet(Module):
    """
    Implements the LeNet architecture for a convolutional neural network.

    This network architecture is a small model that is capable of recognizing patterns
    in images, using alternating layers of convolution and pooling to process spatial data,
    followed by fully connected layers for classification.

    Attributes:
        conv1 (Conv2d): The first convolutional layer.
        relu1 (ReLU): The ReLU activation function after the first convolution.
        maxpool1 (MaxPool2d): The first pooling layer.
        conv2 (Conv2d): The second convolutional layer.
        relu2 (ReLU): The ReLU activation function after the second convolution.
        maxpool2 (MaxPool2d): The second pooling layer.
        fc1 (Linear): The first fully connected layer.
        relu3 (ReLU): The ReLU activation function after the first fully connected layer.
        fc2 (Linear): The second fully connected layer.
        log_soft_max (LogSoftmax): The softmax activation function for output.
    """

    def __init__(self, input_shape: torch.Size, classes: int):
        """Initializes the LeNet model.

        Args:
            input_shape (torch.Size): The shape of the input data.
            classes (int): The number of output classes.
        """
        super().__init__()
        channel_count = input_shape[0]
        self.conv1 = Conv2d(in_channels=channel_count, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        conv_out = int((input_shape[1]-5)/2)+1

        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        conv_out = int((conv_out-5)/2)+1
        conv_size = conv_out * conv_out * 50
        lprint(conv_out)

        self.fc1 = Linear(in_features=conv_size, out_features=500)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.log_soft_max = LogSoftmax(dim=1)

    def forward(self, x_p):
        """Defines the forward pass of the model.

        Args:
            x_p (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output predictions.
        """
        x_p = self.conv1(x_p)
        x_p = self.relu1(x_p)
        x_p = self.maxpool1(x_p)
        x_p = self.conv2(x_p)
        x_p = self.relu2(x_p)
        x_p = self.maxpool2(x_p)
        x_p = flatten(x_p, 1)
        x_p = self.fc1(x_p)
        x_p = self.relu3(x_p)
        x_p = self.fc2(x_p)
        output = self.log_soft_max(x_p)
        return output


#==================================================================================================
#                                       TRAINING WRAPPER
#==================================================================================================

def parse_arguments():
    """Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset_path',
                        type=str,
                        required=True,
                        help="Path to the dataset directory"
                        )
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=32,
                        help="Batch size for training"
                        )
    parser.add_argument('-t',
                        '--train_split',
                        type=float,
                        default=0.7,
                        help="Train/validation split ratio"
                        )
    parser.add_argument('--initial_learning_rate',
                        type=float,
                        default=1e-3,
                        help="Initial learning rate for the optimizer"
                        )
    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help="Number of epochs to train the model"
                        )
    parser.add_argument('-l',
                        '--log_path',
                        type=str,
                        default=None,
                        help="Path to save the log file"
                        )
    parser.add_argument('-m',
                        '--model_output_path',
                        type=str,
                        required=True,
                        help="Path to save the trained model"
                        )
    parser.add_argument('-c',
                        '--csv_output_path',
                        type=str,
                        required=True,
                        help="Path to save the training history CSV")
    return parser.parse_args()


def get_data_loaders(dataset_path: str, train_split: float, batch_size: int):
    """Creates DataLoader objects for the training and validation datasets.

    Args:
        dataset_path (str): Path to the dataset directory.
        train_split (float): The ratio of the dataset to be used for training.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader,
        and the class names.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    img_folder = ImageFolder(dataset_path, transform=transform)
    lprint('Image folder length %d', len(img_folder))
    training_samples_count = int(len(img_folder) * train_split)
    validation_samples_count = len(img_folder) - training_samples_count
    train_data, val_data = random_split(img_folder,
                                        [training_samples_count,
                                         validation_samples_count],
                                        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, validation_loader, img_folder.classes

# pylint: disable=too-many-arguments
class TrainingContext:
    """
    A class to hold and manage the context of a training session.

    This class encapsulates the training configuration including the model, optimizer,
    loss function, and device. It also manages training and validation steps.

    Attributes:
        device (torch.device): The device (CPU/GPU) on which the model will be trained.
        model (Module): The model being trained.
        optimizer (Optimizer): The optimizer used for training.
        loss_fn (Loss): The loss function used for calculating training and validation loss.
        train_steps (int): Number of training steps per epoch.
        validation_steps (int): Number of validation steps per epoch.
    """
    def __init__(self, device, model, optimizer, loss_fn, train_steps, validation_steps):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_steps = train_steps
        self.validation_steps = validation_steps

# pylint: disable=too-many-arguments
def setup_training(device_name,
                   input_shape,
                   classes,
                   initial_learning_rate,
                   train_loader,
                   validation_loader):
    """Setup the training context and return it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lprint("Device -> %s", device_name)
    model = LeNet(input_shape, len(classes)).to(device)
    optimizer = Adam(model.parameters(), lr=initial_learning_rate)
    loss_fn = torch.nn.NLLLoss()
    train_steps = len(train_loader.dataset) // train_loader.batch_size
    validation_steps = len(validation_loader.dataset) // validation_loader.batch_size
    return TrainingContext(device, model, optimizer, loss_fn, train_steps, validation_steps)


def execute_epoch(context, data_loader, train=True):
    """Execute one epoch of training or validation."""
    if train:
        context.model.train()
    else:
        context.model.eval()

    total_loss = 0.0
    correct_predictions = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(context.device), labels.to(context.device)
        outputs = context.model(inputs)
        loss = context.loss_fn(outputs, labels)

        if train:
            context.optimizer.zero_grad()
            loss.backward()
            context.optimizer.step()

        total_loss += loss.item()
        correct_predictions += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    return total_loss, correct_predictions

class TrainingHistory:
    """
    A class that holds and manages the training history for
    a machine learning model during training.

    This class stores the loss and accuracy metrics for both training and validation phases
    and provides methods to update and summarize these metrics.

    Attributes:
        train_loss (list of float): A list that records the training loss after each epoch.
        train_acc (list of float): A list that records the training accuracy after each epoch.
        val_loss (list of float): A list that records the validation loss after each epoch.
        val_acc (list of float): A list that records the validation accuracy after each epoch.
    """

    def __init__(self):
        """
        Initializes the TrainingHistory class with empty lists for training and validation metrics.
        """
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def update(self, train_loss, train_acc, val_loss, val_acc):
        """
        Updates the training history with the latest metrics from an epoch.

        Args:
            train_loss (float): The training loss measured at the end of the epoch.
            train_acc (float): The training accuracy measured at the end of the epoch.
            val_loss (float): The validation loss measured at the end of the epoch.
            val_acc (float): The validation accuracy measured at the end of the epoch.
        """
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

    def print_epoch_summary(self, epoch, epochs):
        """
        Prints a summary of the training metrics for a specific epoch.

        Args:
            epoch (int): The current epoch number (zero-indexed).
            epochs (int): The total number of epochs for which the model is being trained.
        """
        print(f"[INFO] EPOCH: {epoch + 1}/{epochs}")
        print(f"Train loss: {self.train_loss[-1]:.6f}, Train accuracy: {self.train_acc[-1]:.4f}")
        print(f"Val loss: {self.val_loss[-1]:.6f}, Val accuracy: {self.val_acc[-1]:.4f}\n")

# pylint: disable=too-many-locals
def train_network(initial_learning_rate: float,
                  epochs: int,
                  train_loader: DataLoader,
                  validation_loader: DataLoader,
                  classes: list[str]):
    """Trains the neural network."""
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    input_shape = next(iter(train_loader))[0][0].shape
    context = setup_training(device_name,
                             input_shape,
                             classes,
                             initial_learning_rate,
                             train_loader,
                             validation_loader
                             )
    history = TrainingHistory()

    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_correct = execute_epoch(context, train_loader, train=True)
        val_loss, val_correct = execute_epoch(context, validation_loader, train=False)

        avg_train_loss = train_loss / context.train_steps
        avg_val_loss = val_loss / context.validation_steps
        # train_accuracy = train_correct / len(train_loader.dataset)
        # val_accuracy = val_correct / len(validation_loader.dataset)
        train_accuracy = train_correct / len(cast(Sized, train_loader.dataset))
        val_accuracy = val_correct / len(cast(Sized, validation_loader.dataset))

        history.update(avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)
        history.print_epoch_summary(epoch, epochs)

    end_time = time.time()
    print(f"[INFO] total time taken to train the model: {end_time - start_time:.2f}s")
    return context.model, history


def save_model(model, path):
    """Saves the trained model to a file.

    Args:
        model (torch.nn.Module): The trained model.
        path (str): The path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")

def save_history_to_csv(history, path):
    """Saves the training history to a CSV file.

    Args:
        history (TrainingHistory): The training history object.
        path (str): The path to save the CSV file.
    """
    with open(path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i, (train_loss, train_acc, val_loss, val_acc) in enumerate(
                        zip(history.train_loss,
                            history.train_acc, history.val_loss, history.val_acc)):
            writer.writerow([i + 1, train_loss, train_acc, val_loss, val_acc])
    print(f"[INFO] Training history saved to {path}")


def visualisation_of_history(history):
    """Visualizes the training and validation accuracy.

    Args:
        history (TrainingHistory): The training history object.
    """
    plt.title('Accuracy')
    plt.plot(history.train_acc, '-', label='Train')  # Dostęp do atrybutu klasy
    plt.plot(history.val_acc, '--', label='Validation')  # Dostęp do atrybutu klasy
    plt.legend()
    plt.show()

def main(args):
    """Main function to set up the logger, train the model, and save outputs.

    Args:
        args (argparse.Namespace): The parsed command line arguments.
    """
    setup_logger(args.log_path)
    train_loader, validation_loader, classes = get_data_loaders(args.dataset_path,
                                                                args.train_split,
                                                                args.batch_size)
    model, history = train_network(args.initial_learning_rate,
                                   args.epochs,
                                   train_loader,
                                   validation_loader,
                                   classes)
    visualisation_of_history(history)
    save_model(model, args.model_output_path)
    save_history_to_csv(history, args.csv_output_path)

if __name__ == '__main__':
    main(parse_arguments())
