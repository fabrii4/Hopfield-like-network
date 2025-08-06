# Hopfield-like network

## Work in progress!

As briefly discussed in [Hopfield-like network.pdf](./Hopfield-like%20network.pdf), 
an Hopfield-like feed-forward network is capable to store and retrieve an 
indefinite number of patterns (given by the number of output neurons).

A simple proof of concept of such a network is provided in [HN_minst_test.py](./HN_minst_test.py).
In this example, 10 samples from the MINST dataset (1 for each class) are stored in a 1-layer Hopfield-like network.
When 10 different samples from the same dataset are run throught the network, they are correctly classified most of the time.

A simple improvement on this approach would be to consider a 2-layers architecture where multiple samples for each class are stored in the first layer and assigned to the same output neuron in the second layer.   
An example of this approach is given in [hopfield_with_encoder.py](./tests/hopfield_with_encoder.py) where patterns from the MNIST dataset are first encoded by a convolutional encoder and then stored in a 2-layers Hopfield network.
