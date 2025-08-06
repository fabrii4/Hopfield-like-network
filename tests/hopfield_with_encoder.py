import sys
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from autoencoder import Autoencoder
import train_functions as tf

device = torch.device('cuda')
#device = torch.device('cpu')

#convert to binary representation
def dec2bin(x, L, bits=8):
    Nb=2**bits-1
    x=(((x+L)/(2*L))*Nb).round().long()
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    x = x.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    x = 2*x-1
    return x
    
def bin2dec(b, L, bits=8):
    Nb=2**bits-1
    mask = 2**torch.arange(bits).to(b.device, b.dtype)
    x = torch.sum(mask * b, -1)
    x=L*(2*x/Nb-1)
    return x



class Hopfield(nn.Module):
    def __init__(self, dataloader, use_encoder=True):
        super().__init__()
        self.n_class=10
        self.L=30
        self.nbits=8
        self.use_encoder = use_encoder
        self.flatten = nn.Flatten()
        self.W = torch.tensor(0)
        self.W1 = torch.tensor(0)
        self.autoencoder = Autoencoder()
        self.autoencoder.load_state_dict(torch.load("./weights/autoenc_model.tar"))
        self.init_hopfield_layers(dataloader)

        
    def forward(self, x):
        with torch.no_grad():
            x_enc = self.autoencoder.encode(x) if self.use_encoder else self.flatten(x)
        #convert to binary representation
        x_enc = dec2bin(x_enc[0], self.L, self.nbits).float()
        x_enc = torch.permute(x_enc,(1,0))
        x = torch.matmul(self.W, x_enc.unsqueeze(-1)).squeeze(-1)
        x = torch.sum(x,0)
        x = torch.where(x == x[torch.argmax(x)], 1.,0.)
        x = torch.matmul(self.W1, x.unsqueeze(-1)).squeeze(-1)
        x = torch.argmax(x)
        return x, x_enc
        
    #extract classes representative from dataset to initialize the network
    def init_hopfield_layers(self,dataloader):
        class_sample=[]
        i_class=0
        for x, y in dataloader:
            if i_class == self.n_class:
                break
            if y[0] != i_class:
                continue
            i_class+=1
            with torch.no_grad():
                x_enc = self.autoencoder.encode(x) if self.use_encoder else self.flatten(x)
            x = dec2bin(x_enc[0], self.L, self.nbits)
            class_sample.append(x)
        class_sample=np.array(class_sample)
        #network architecture, initialize weights
        #input_dim=len(class_sample[0])
        #w=w[hidden_dim,input_dim]       h_i = w_ij * s_j
        self.W=torch.tensor(class_sample)
        #w1=w1[n_class,hidden_dim]    o_l = w1_li * h_i
        self.W1=torch.eye(self.n_class)
        self.W = self.W.to(device).float()
        self.W = torch.permute(self.W,(2,0,1))
        self.W1 = self.W1.to(device)
   
    #training procedure
    def learn(self, x, y):
        #run network
        result, x_enc = self.forward(x)
        #if not correctly classified add sample to network
        if result != y:
            self.W = torch.cat((self.W, x_enc.unsqueeze(1)), axis=1)
            self.W1 = torch.cat((self.W1, torch.zeros((self.n_class,1)).to(device)),axis=-1)
            self.W1[y,-1]=1

############################

use_encoder=True

weights_path="./weights/hopfield_model_enc.tar" if use_encoder else "./weights/hopfield_model.tar"

#load datasets (use only half for training)
train_loader, test_loader = tf.load_MNIST_dataset(batch_size=1, split=1)

#init model
model = Hopfield(train_loader, use_encoder)
model.to(device)

print("Hopfield model uses convolutional encoder", use_encoder)
        
#train the model        
if len(sys.argv)>1 and sys.argv[1] == 'train':
    print("train")
    tf.train_hopfield(model, train_loader)  
    print(f"W size: {model.W.shape}, W1 size: {model.W1.shape}")
    #save model
    torch.save(model, weights_path)

#load and test
try:
    #model.load_state_dict(torch.load("hopfield_model.tar"))
    model = torch.load(weights_path, weights_only=False)
    print("model loaded")
except:
    print("failed to load model")
    exit()
print(f"W size: {model.W.shape}, W1 size: {model.W1.shape}")


#evaluate
correct=0
print("test")
N_test=len(test_loader.dataset)
for x, y in tqdm.tqdm(test_loader.dataset):
    p, _ = tf.run_model(model, x)
    correct += p==y
print(f"Test accuracy: {correct/N_test:.3f}")


##backpropagate categories to visualize stored patters
#from matplotlib import pyplot as plt
#digits=[0,1,2,3,4,5,6,7,8,9]
#for digit in digits:
#    y = F.one_hot(torch.tensor(digit), num_classes=10)
#    y = y.expand(model.nbits,-1,).to(device)
#    W1 = torch.permute(model.W1,(0,2,1))
#    W = torch.permute(model.W,(0,2,1))
#    y = torch.matmul(W1,y.unsqueeze(-1).float())
#    y = torch.matmul(W,y).squeeze(-1)
#    y = torch.permute(y, (1,0))
#    y = bin2dec(y, model.L, bits=model.nbits)
#    with torch.no_grad():
#        y = model.autoencoder.decode(y.unsqueeze(0))
#    y=torch.permute(y[0],(1,2,0))
#    f, axarr = plt.subplots(1,1)
#    axarr.imshow(y.cpu(), cmap='gray')
#    plt.title(digit)
#    plt.show()

