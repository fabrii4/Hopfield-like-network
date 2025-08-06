# Hopfield-like network on top of convolutional layers

Here, We apply a 2 layers hopfield network on top of a convolutional encoder.
The convolutional encoder is pretrained separately as part of an autoencoder.

We compare the results with that of 2 other networks:
- a 2 linear layers network on top of the same pretrained convolutional encoder (where only the linear layers are trained) 
- a full convolutional network given by the convolutional encoder + 2 linear layers (all layers trained from scratch)

We test the performance of the 3 networks on the MNIST dataset.

The full CNN achieves the best results (accuracy 99%) followed by the 2 linear layers (accuracy 97.5%) and the 2 layers Hopfield network (accuracy 86.5%)

We expected that the convolutional encoder would smooth out the differences between category elements so that the Hopfield network could more easily associate the inputs to the stored category patterns.
However, this does not seem to be the case: when training on the 60000 samples, the network still stores  around 10000 different patterns (instead of only 10 corresponding to the MNIST categories).
When running the same network without encoder the number of stored patterns is only slightly higher (around 11500) and the accuracy only slightly lower (84.2%).
We need to investigate if it is possible to improve the network performance by using different kind of encoders.

To test the pretrained models simply run
```bash
./run_tests.sh
```

To retrain the models:
```bash
python train_autoencoder.py train
python cnn.py train
python linear_with_encoder.py train
python hopfield_with_encoder.py train
```

The scripts require torch, torchvision, numpy, tqdm:
```bash
pip install torch
pip install torchvision
pip install numpy
pip install tqdm
```
