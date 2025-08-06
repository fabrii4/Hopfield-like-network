import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import train_functions as tf

device = torch.device('cuda')
#device = torch.device('cpu')



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (25, 2, 2))
        self.pool = nn.MaxPool2d(2)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(1,100,3,2)
        self.conv2 = nn.Conv2d(100,50,3,2)
        self.conv3 = nn.Conv2d(50,25,4,2)
        self.act_fn = nn.LeakyReLU(0.01)
        self.act_fn_out = nn.Softmax(dim=1)
        self.L0 = nn.Linear(100, 50)
        self.L1 = nn.Linear(50, 10)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act_fn(x)
        #x = self.pool(x)
        x = self.conv2(x)
        x = self.act_fn(x)
        #x = self.pool(x)
        x = self.conv3(x)
        x = self.act_fn(x)
        #x = self.pool(x)
        x = self.flatten(x)
        x = self.L0(x)
        x = self.act_fn(x)
        x = self.L1(x)
        print(x)
        x = self.act_fn_out(x)
        print(x)
        exit()
        return x

        
############################

weights_path="./weights/cnn_model.tar"

#init model
model = CNN()
model.to(device)
summary(model,(1,32,32))
loss_module = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

#load datasets
train_loader, test_loader = tf.load_MNIST_dataset()

        
#train the model        
if len(sys.argv)>1 and sys.argv[1] == 'train':            
    try:   
        model.load_state_dict(torch.load(weights_path))
        print("weights loaded")
    except:
        print("failed to load weights")
        pass
    print("train")
    tf.train_model(model, optimizer, train_loader, loss_module, num_epochs=30)      
    #save model
    state_dict = model.state_dict()
    #print(state_dict)
    torch.save(state_dict, weights_path)

#load and test
try:
    model.load_state_dict(torch.load(weights_path))
    print("weights loaded")
except:
    print("failed to load weights")
    exit()

#evaluate
correct=0
N_test=len(test_loader.dataset)
print("test")
for x, y in test_loader.dataset:
    p = tf.run_model(model, x)
    p = torch.argmax(p).item()
    correct += p==y
print(f"Test accuracy: {correct/N_test:.3f}")

