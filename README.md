# Tomato-Plant-Disease-Detection-Using-Deep-Learning-Early-Blight-Classification
 Tomato Disease Classification This deep learning project classifies tomato plant health into two categories: Tomato_Early_blight and Tomato_healthy. Built-in Python with Jupyter Notebook, the model is trained on Kaggle data using CNNs for accurate disease detection, aiming to help farmers with early blight identification and timely interventions.

# Project Structure
data/: Contains the dataset with labeled images for training and testing.
Notebooks/: Jupyter notebooks are used for data preprocessing, model training, evaluation, and visualization of results.
models/: Saved models and checkpoints for reproducibility.
README.md: Project overview and instructions.

# Data Description
The dataset used in this project consists of images categorized into two classes:

Tomato_Early_blight: Images of tomato leaves affected by early blight.
Tomato_healthy: Images of healthy tomato leaves.
Model Architecture
This project leverages Convolutional Neural Networks (CNNs) due to their effectiveness in image recognition tasks. Techniques like data augmentation, dropout, and batch normalization are applied to improve model accuracy and reduce overfitting.

# Training & Evaluation
The model is trained using standard metrics such as accuracy and loss, with a validation set to monitor performance. Hyperparameter tuning and optimization techniques have been used to achieve robust classification results.

# Results
The final model demonstrates strong classification performance, achieving high accuracy in distinguishing between healthy and diseased tomato leaves.

# Getting Started
Clone the repository and install dependencies.
Download the dataset from Kaggle and place it in the data/ directory.
Run the Jupyter notebooks to train and evaluate the model.

# Future Work
Potential future improvements include expanding the dataset to incorporate additional diseases, experimenting with different model architectures, and deploying the model as a web application for real-time disease detection.

