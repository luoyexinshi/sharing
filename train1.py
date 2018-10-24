import argparse

__author__ = 'Luoyexin Shi'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu') ## how to access it??
parser.add_argument('--arch', type=str, default='densenet121', help='architecture [available: densenet, vgg]', required=True)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=str, default='500', help='hidden units for fc layers (comma separated)')
parser.add_argument('--epochs', type=int, default=4, help='number of epochs')
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
parser.add_argument('--saved_model_path' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
args = parser.parse_args()

gpu = args.gpu
arch = args.arch
lr = args.lr
hidden_units = args.hidden_units
epochs = args.epochs
data_dir = args.data_dir
cat_to_name = args.cat_to_name
saved_model_path = args.saved_model_path

## training data
# Imports here

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

data_dir = args.data_dir
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


# Train a model with a pre-trained network
model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict((
                          ('fc1', nn.Linear(1024, 500),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1)))))

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
