# easy-wgan
using pytorch 2.0+ and mayplotlib
The discriminator in WGAN is different from that in GAN. The sigmoid function should not be added to the last layer of the discriminator; otherwise, the g_loss will approach -1 in the first batch, making it impossible to update the gradient.
# WGAN with Gradient Penalty
This repository contains an implementation of a Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty (GP) using PyTorch. The model is trained on the CIFAR-10 dataset.
Model Architecture
Generator: The generator network takes a 100-dimensional noise vector and generates 224x224 RGB images through a series of transposed convolutional layers.
Discriminator: The discriminator network classifies 224x224 RGB images as real or fake through a series of convolutional layers.
Training
The training process involves alternating between updating the discriminator and the generator. The discriminator is updated more frequently to ensure it learns to distinguish real images from generated ones effectively. Gradient penalty is applied to enforce the Lipschitz constraint.
Usage
Install dependencies:
pip install torch torchvision matplotlib
Download CIFAR-10 dataset: The dataset will be automatically downloaded when you run the training script.
Train the model:
python CIFAR10-WGAN-2（梯度惩罚）.py
Results
Generated images are displayed every 10 epochs to monitor the training progress.
Code
Python
