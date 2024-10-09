# Cats vs Dogs - Machine Learning Project

**Hey there! I'm Sri Chaitanya 👋**

In this project, I built a neural network using **TensorFlow** and **Keras** to classify images as either dogs or cats.

## Dataset

I utilized a dataset from Kaggle for training and testing my model. You can find the dataset [here](https://www.kaggle.com/datasets/d4rklucif3r/cat-and-dogs?select=dataset).

## Project Overview

The objective of this project is to create a model that can accurately identify whether a given image contains a dog or a cat. Below are the steps I followed during the development of this project:

1. **Environment Setup**:
   - Installed necessary libraries such as TensorFlow and Keras.
   - Set up the working environment for model training.

2. **Data Preprocessing**:
   - Used `ImageDataGenerator` to augment the training data by rescaling and applying transformations like shear, zoom, and horizontal flip.
   - Split the dataset into training and testing sets for effective evaluation.

3. **Model Architecture**:
   - Constructed a Convolutional Neural Network (CNN) with the following layers:
     - Convolutional layers to extract features from images.
     - Max Pooling layers to reduce dimensionality and prevent overfitting.
     - Flattening layer to convert 2D feature maps to 1D feature vectors.
     - Dense layers for high-level reasoning, concluding with a sigmoid output for binary classification.

4. **Model Training**:
   - Compiled the model with the Adam optimizer and binary cross-entropy loss function.
   - Trained the model on the training dataset while validating it on the test dataset for 20 epochs.

5. **Making Predictions**:
   - Loaded a sample image and preprocessed it to match the input requirements of the CNN.
   - Made predictions and classified the image as either a dog or a cat based on the model's output.

## Conclusion

This project was a great opportunity to dive into the world of machine learning and explore the applications of CNNs in image classification. I learned a lot about data preprocessing, model architecture, and evaluation metrics.

Feel free to explore the code and datasets! If you have any questions or suggestions, please reach out.

## Acknowledgements

- [Kaggle](https://www.kaggle.com) for providing the dataset.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for their excellent libraries and documentation.

