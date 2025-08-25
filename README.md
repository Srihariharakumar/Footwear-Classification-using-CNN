# Footwear-Classification-using-CNN
This project implements a Convolutional Neural Network (CNN) to classify real-life footwear images into three categories:

 Shoes

 Sneakers

 Slippers

The dataset was created using images captured with a mobile phone, ensuring real-world applicability.
Dataset

Total Images: 90 (30 per class)

Structure:
Footwear_data/
 ├── Shoes/
 │    ├── shoe1.jpg
 │    ├── shoe2.png
 │    └── ...
 ├── Sneakers/
 │    ├── sneaker1.jpg
 │    ├── sneaker2.png
 │    └── ...
 ├── Slippers/
      ├── slipper1.jpg
      ├── slipper2.png
      └── ...
Requirements

Install required libraries before running:
pip install tensorflow matplotlib numpy

Project Workflow

Data Preprocessing

Images resized to 128x128

Normalization (rescale=1./255)

Data Augmentation (rotation, zoom, horizontal flip)

Train/Validation Split = 80/20

CNN Architecture

Convolutional Layers: (32, 64, 128 filters)

MaxPooling layers

Fully Connected Dense layer (128 neurons, ReLU)

Dropout for regularization

Output Layer: 3 neurons (Softmax)

Training

Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 20

Results

Model trained successfully with increasing accuracy.
Final Validation Accuracy: 93.75%
