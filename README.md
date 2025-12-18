# Chest X-ray AI 🩺 
<img width="960" height="375" alt="image" src="https://github.com/user-attachments/assets/b50ab192-d3b7-4554-b305-e7f1b8303bd8" />


  
---
## Project Overview
This project is an AI-powered tool for detecting pneumonia from chest X-ray images.  
We apply **Singular Value Decomposition (SVD)** as preprocessing to compress images, then classify using a **Convolutional Neural Network (CNN)**.  
A simple explainable AI (XAI) component gives users a layman-friendly interpretation of the results.
---
## Objectives of the Project:
To develop an automated system for detecting pneumonia from chest X-ray images using deep learning,
To improve diagnostic speed and consistency compared to manual interpretation,
To evaluate the effectiveness of a CNN-based classification model
## Model Training 
Input Size: 224 × 224 grayscale chest X-ray images,
Three convolutional layers with  ReLu activation function,
Max-pooling layers, 
Fully connected layer, 
Dropout layer (rate = 0.5), 
Output Layer: Sigmoid activation for binary classification (Normal / Pneumonia),
Optimizer: Adam,
Learning Rate: 0.0001,
Loss Function: Binary Cross-Entropy,
Batch Size: 16
## Evaluation matrics
Accuracy,
Precision,
Recall,
F1-Score,
Confusion matrix
## Features
- Upload chest X-ray images (`.png`, `.jpg`, `.jpeg`)  
- AI prediction: Normal vs Pneumonia  
- Confidence score of the prediction  
- Layman explanation with one-click chatbox
---

# Install dependencies
pip install -r requirements.txt
