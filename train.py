import argparse

__author__ = 'Luoyexin Shi'


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Script training data from a data directory')
    # Add arguments
    parser.add_argument(
        '-d', '--data_directory', type=str, help='Data Directory', required=True)

    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    data_dir = args.data_dir
    # Return all variable values
    return data_dir

# Run get_args()
# get_args()

# Match return values from get_arguments()
# and assign to their respective variables
data_dir = get_args()


## training data
# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from collections import OrderedDict

import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models

import tensorflow as tf

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
# Label mapping
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Train a model with a pre-trained network
model = models.densenet121(pretrained=True)
model

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# Defining criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    model.to('cuda')
    for ii, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.cuda.FloatTensor).mean()

    return test_loss, accuracy

# Model training, printing loss, test loss, test accuracy
def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                     "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                     "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                running_loss = 0


            # Make sure training is back on
                model.train()

# keep active session
from workspace_utils import active_session
with active_session():
    do_deep_learning(model, trainloader, 4, 200, criterion, optimizer, 'gpu')

# Save the checkpoint
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())
# TODO: Save the checkpoint
checkpoint = {'state_dict': model.state_dict(),
             'classifier': model.classifier,
             'arch_name':'densenet121',
             'class_idx': train_data.class_to_idx,
             'epoch': 4,
             'optimizer': optimizer.state_dict()}

model.class_to_idx = train_data.class_to_idx
torch.save(checkpoint, 'checkpoint.pth')
torch.save(optimizer.state_dict(), 'optimizer.pth')
