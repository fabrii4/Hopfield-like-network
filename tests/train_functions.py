import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision import transforms

device = torch.device('cuda')
#device = torch.device('cpu')



#training function for Hopfield model
def train_hopfield(model, data_loader):
    # Training loop
    #n_steps=0
    #n_max=2
    for data_inputs, data_labels in tqdm.tqdm(data_loader):
        ## Step 1: Move input data to device (only strictly necessary if we use GPU)
        data_inputs = data_inputs.to(device)
        data_labels = data_labels.to(device)
        ## Step 2: Run the model on the input data
        model.learn(data_inputs, data_labels[0])
        #n_steps+=1
        #if n_steps>n_max:
        #    break


#training function
def train_model(model, optimizer, data_loader, loss_module, num_epochs=100, cat=True):
    # Set model to train mode
    model.train()
    N_train = len(data_loader.dataset)
    # Training loop
    for epoch in range(num_epochs):
        loss_epoch, correct, N_step = 0, 0, 0
        for data_inputs, data_labels in tqdm.tqdm(data_loader):
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            data_labels_cat = F.one_hot(data_labels, num_classes=10)
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            ## Step 3: Calculate the loss
            #use data_inputs when training autoencoder
            true = data_labels_cat.float() if cat else data_inputs
            loss = loss_module(preds, true)
            ## Step 4: Perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            ## Step 5: Update the parameters
            optimizer.step()
            #calculate loss and accuracy
            loss_epoch+=loss.item()
            N_step+=1
            if cat:
                correct += (torch.argmax(preds, dim=1) == data_labels).sum()
        loss_epoch/=N_step
        accuracy = correct / N_train
        print(f"Epoch {epoch}, Loss: {loss_epoch}, Accuracy: {accuracy:.3f}")
            
#run model
def run_model(model, data_input):
    model.eval()
    with torch.no_grad(): # Deactivate gradients for the following code
        # Determine prediction of model on dev set
        data_input = data_input.to(device)
        data_input = torch.unsqueeze(data_input, 0)
        preds = model(data_input)
        return preds
        
        
#testing function
def test_model(model, test_loader, loss_module):
    model.eval()
    with torch.no_grad():
        loss_epoch = 0
        for data_inputs, data_labels in test_loader:
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            data_labels_cat = F.one_hot(data_labels, num_classes=10)
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_inputs)
            #calculate loss and accuracy
            loss_epoch+=loss.item()
        print(f"Test Loss: {loss_epoch}")
        
        
#dataset function
def load_MNIST_dataset(batch_size=128, split=1):
    #Dataset
    transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()])#, transforms.Normalize((0.1309,), (0.2821,))])
    dataset = MNIST(root="./datasets", train=True, download=True, transform=transform)
    #split to train and validation datasets
    if split<1:
        N_train = int(len(dataset)*split)
        N_val = len(dataset) - N_train    
        dataset, val_set = data.random_split(dataset, [N_train, N_val])
    #train and validation dataloaders
    train_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    #val_loader = data.DataLoader(val_set, batch_size=batch_size, num_workers=4)
    #get test dataset
    dataset_test = MNIST(root="./datasets", train=False, download=True, transform=transform)
    test_loader = data.DataLoader(dataset_test, batch_size=batch_size, num_workers=4)
    return train_loader, test_loader

