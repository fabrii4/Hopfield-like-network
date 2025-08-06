import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from autoencoder import Autoencoder
import train_functions as tf

device = torch.device('cuda')
#device = torch.device('cpu')



class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = nn.LeakyReLU(0.01)
        self.act_fn_out = nn.Softmax(dim=1)
        self.L0 = nn.Linear(100, 50)
        self.L1 = nn.Linear(50, 10)
        self.autoencoder = Autoencoder()
        self.autoencoder.load_state_dict(torch.load("./weights/autoenc_model.tar"))

        
    def forward(self, x):
        with torch.no_grad():
            x = self.autoencoder.encode(x)
        x = self.L0(x)
        x = self.act_fn(x)
        x = self.L1(x)
        x = self.act_fn_out(x)
        return x
        

########################

weights_path="./weights/linear_model.tar"

#init model
model = Linear()
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
print("test")
N_test = len(test_loader.dataset)
for x, y in test_loader.dataset:
    p = tf.run_model(model, x)
    p = torch.argmax(p).item()
    correct += p==y
print(f"Test accuracy: {correct/N_test:.3f}")

