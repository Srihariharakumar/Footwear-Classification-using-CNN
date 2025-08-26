# CNN Real-Life Footwear Classification

This project implements a **Convolutional Neural Network (CNN)** to classify real-life footwear images into three categories:
- Shoes
- Sneakers
- Slippers

The dataset was created using images captured with a mobile phone, ensuring real-world applicability.

---

## Dataset
- Total Images: **90** (30 per class)  
- Classes: `Shoes`, `Sneakers`, `Slippers`

Folder structure:
CNN_realife/
 - Shoes/shoe1.jpg/shoe2.png...
 - Sneakers/sneaker1.jpg/sneaker2.png...
 - Slippers/slipper1.jpg/slipper2.png...

## Project Workflow

## 1. Data Preprocessing

Images resized to 128x128

Normalized pixel values (rescale=1./255)

Data Augmentation (rotation, zoom, horizontal flip)

Train/Validation Split = 80/20

## 2.CNN Model Architecture

Convolutional Layers: (32, 64, 128 filters)

MaxPooling layers after each convolution

Flatten → Dense(128, ReLU)

Dropout(0.5) for regularization

Output Layer: Dense(3, Softmax)

## 3.Training

Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 20

## Results

 - The model was successfully trained with the accuracy of 93.5%

## Prediction Example

Use the trained model to predict any new image:

from tensorflow.keras.preprocessing import image
import numpy as np
img_path = "CNN_realife/Sneakers/sneaker1.jpg"  
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
class_labels = list(train_data.class_indices.keys())
print("Predicted Class:", class_labels[np.argmax(prediction)])
model.save("cnn_shoe_classifier.h5")

