#!/bin/bash
echo ""
echo "-------------------------"
echo "CNN"
python cnn.py
echo ""
echo "-------------------------"
echo "2 linear layers with encoder"
python linear_with_encoder.py
echo ""
echo "-------------------------"
echo "2 linear Hopfield with encoder"
python hopfield_with_encoder.py
