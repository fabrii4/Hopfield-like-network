import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from autoencoder import Autoencoder
import train_functions as tf

device = torch.device('cuda')
#device = torch.device('cpu')


weights_path="./weights/autoenc_model.tar"

#init model
model = Autoencoder()
model.to(device)
loss_module = nn.MSELoss()
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
    tf.train_model(model, optimizer, train_loader, loss_module, num_epochs=100, cat=False)      
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

print("test")
tf.test_model(model, test_loader, loss_module)    

#evaluate
from matplotlib import pyplot as plt
for x, y in test_loader.dataset:
    p = tf.run_model(model, x)
    x = x.unsqueeze(0)
    p=torch.permute(p[0],(1,2,0))
    x=torch.permute(x[0],(1,2,0))
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(x.cpu(), cmap='gray')
    axarr[1].imshow(p.cpu(), cmap='gray')
    plt.title(y)
    plt.show()


