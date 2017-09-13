# %load teacher-student.py
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from student import *
from teacher import *

teacher = None
student = None
epochs = 5
batch_size = 64
lr = 1e-3

optimizer = None

def train():
    T = 17
    student.train()
    for p in teacher.parameters():
        p.requires_grad = False
    for epoch in xrange(epochs):
        avg_loss = 0
        n_batches = len(transfer_loader)
        for data, _ in transfer_loader:
            data = data.cuda()
            data = Variable(data)
            optimizer.zero_grad()
            output_teacher = F.softmax(teacher(data) / T)
            output_student = F.softmax(student(data) / T)
            loss = F.binary_cross_entropy(output_student, output_teacher)
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]
            
        avg_loss /= n_batches
        print(avg_loss)
        
    
def test():
    T = 1
    student.eval()
    correct = 0
    for data, label in test_loader:
        data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label)
        output = F.log_softmax(student(data) / T)
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    
    print(100. * correct / len(test_loader.dataset))

student = Student()
student.cuda()
with open('teacher.params', 'rb') as f:
    teacher = torch.load(f)

optimizer = optim.Adam(student.parameters(), lr=lr)

train_set = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

transfer_data, transfer_labels = [], []
for data, label in train_set:
    if label != 3:
        transfer_data.append(data.tolist())
        transfer_labels.append(label)
        
transfer_data, transfer_labels = torch.Tensor(transfer_data), torch.Tensor(transfer_labels)

kwargs = {'num_workers': 1, 'pin_memory': True}
transfer_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(transfer_data, transfer_labels),
        batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
        batch_size=batch_size, shuffle=True, **kwargs)

print(teacher)

teacher.eval()
train()
test()