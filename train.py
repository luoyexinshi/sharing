#! /usr/bin/env python
import argparse

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
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)



def run(args):
	filename = args.input # these match the "dest": dest="input"
	output_filename = args.output # from dest="output"
	qual = args.quality_score # default is I

	# Do stuff


def main():
    parser=argparse.ArgumentParser(description="Convert a fastA file to a FastQ file")
    parser.add_argument("-in",help="fasta input file" ,dest="input", type=str, required=True)
    parser.add_argument("-out",help="fastq output filename" ,dest="output", type=str, required=True)
    parser.add_argument("-q",help="Quality score to fill in (since fasta doesn't have quality scores but fastq needs them. Default=I" ,dest="quality_score", type=str, default="I")
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main()
