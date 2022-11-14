import numpy as np
from PIL import Image


#import dataset
train_set=[]
path="./minst/train/"
thresh=25
for i in range(10):
    image=Image.open(path+str(i)+".png")
    #convert image to binary numpy array (s=+/-1)
    data = np.asarray(image).flatten()
    data=np.where(data > thresh, 1, -1)
    train_set.append(data)
train_set=np.array(train_set)


##store patterns in hopfield like network
w=np.transpose(train_set)


#test

#import dataset
test_set=[]
path="./minst/test/"
thresh=25
for i in range(10):
    image=Image.open(path+str(i)+".png")
    #convert image to binary numpy array (s=+/-1)
    data = np.asarray(image).flatten()
    data=np.where(data > thresh, 1, -1)
    test_set.append(data)


#run the network on the test set
print("Test Results:")
for i in range(10):
    output=np.dot(test_set[i],w)
    print("Test element: "+str(i)+" --> Test result: "+str(np.argmax(output)))
    #convert output to percentage
    print("Output: "+str((output*100)//784))

