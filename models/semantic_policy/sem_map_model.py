import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from utils.model import get_grid, ChannelPool, Flatten, NNBase

import cv2
import time

class UNet(nn.Module):
    
    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16): #input shape is (240, 240)

        super(UNet, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 *15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.deconv_main = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(1, 1, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(1, 1, 3, stride=1, padding=2),
            nn.ReLU()
        )

        self.linear1 = nn.Linear(out_size * 32 + 256, hidden_size) #outsize is 15^2 (7208 total)
        self.linear2 = nn.Linear(hidden_size, 256)
        #self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(73, 256) #73 object categories
        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.train()

    def forward(self, inputs, goal_cats): 
        x = self.main(inputs)
        #print("x shape is ", x.shape)
        #orientation_emb = self.orientation_emb(extras[:,0]) 
        goal_emb = self.goal_emb(goal_cats).view(-1,256) #goal name
        #print("goal emb shape is ", goal_emb.shape)

        x = torch.cat((x, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))

        x = x.view(-1, 1, 16,16)
        x = self.deconv_main(x) #WIll get Nx1x8x8
        x = self.flatten(x)
        #x = self.softmax(x)
        return x


class UNetMulti(nn.Module):
    
    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16): #input shape is (240, 240)

        super(UNetMulti, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 *15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 73, 3, stride=1, padding=1),
            #nn.Conv2d(64, 32, 3, stride=1, padding=1),
            #nn.ReLU(),
            #Flatten()
        )

        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.train()

    def forward(self, inputs): 
        x = self.main(inputs)
        #x = self.flatten(x)
        #x = self.softmax(x)
        return x

class UNetDot(nn.Module):
    
    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16): #input shape is (240, 240)

        super(UNet, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 *15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, stride=1, padding=1),
            nn.ReLU()
            #nn.Conv2d(64, 32, 3, stride=1, padding=1),
            #nn.ReLU(),
            #Flatten()
        )



        self.deconv_main = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=1),
            nn.ReLU(),
            #nn.AvgPool2d(2),
            nn.Conv2d(128, 64, 1, stride=1, padding=1),
            nn.ReLU(),
            #nn.AvgPool2d(2),
            nn.Conv2d(64, 32, 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1, stride=1, padding=1),
            nn.ReLU(),
        )

        #self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size) #outsize is 15^2 (7208 total)
        #self.linear2 = nn.Linear(hidden_size, 256)
        #self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(73, 256) #73 object categories
        self.goal_lin = nn.Linear(256, 128)
        self.goal_lin2 = nn.Linear(128, 128)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.train()

    def forward(self, inputs, goal_cats): 
        x = self.main(inputs)
        #print("x shape is ", x.shape)
        #orientation_emb = self.orientation_emb(extras[:,0]) 
        goal_emb = self.goal_emb(goal_cats).view(-1,256) #goal name
        goal_emb = self.goal_lin(goal_emb)
        goal_emb = self.relu(goal_emb)
        goal_emb = self.goal_lin2(goal_emb)
        goal_emb = self.relu(goal_emb)

        #Tile goal_emb


        #print("goal emb shape is ", goal_emb.shape)

        x = torch.cat((x, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))

        x = x.view(-1, 1, 16,16)
        x = self.deconv_main(x) #WIll get Nx1x8x8
        x = self.flatten(x)
        #x = self.softmax(x)
        return x