# %load student.py
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable

class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.n_inputs = 28 * 28
        self.n_layer_1 = 800
        self.n_layer_2 = 800
        self.n_classes = 10
        self.drop = nn.Dropout(0.5)
        self.affine1 = nn.Linear(self.n_inputs, self.n_layer_1)
        self.affine2 = nn.Linear(self.n_layer_1, self.n_layer_2)
        self.affine3 = nn.Linear(self.n_layer_2, self.n_classes)
    
    def forward(self, x):
        x = x.view(-1, self.n_inputs)
        out1 = self.drop(F.relu(self.affine1(x)))
        out2 = self.drop(F.relu(self.affine2(out1)))
        out3 = self.affine3(out2)
        return out3