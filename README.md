# Mnist-lenet-cuda-accelerator
This repository contains an optimized CUDA implementation of the convolutional layers used in a modified LeNet-5 architecture.

This project implements an optimized CUDA-based forward pass for layers C1 and C3 in a modified LeNet-5 architecture, utilizing the Mini-DNN framework. It processes the Fashion MNIST dataset, consisting of 10,000 single-channel images (86x86 pixels) and outputs classification probabilities for 10 categories. High-performance inference is achieved using CUDA optimization techniques such as:
1)Tensor Cores for mixed-precision computations
2)Kernel fusion to minimize overhead
3)loop unrolling 
4)CUDA streams
5)Fp-16 arithmetic and other acceleration techniques..
