# Maize Leaf Disease Classification using CNN Architectures

This repository contains multiple Jupyter notebooks implementing deep learning models for classifying maize leaf diseases. The models use popular Convolutional Neural Network (CNN) architectures pre-trained on ImageNet and fine-tuned for the task.

## Project Structure

Each notebook corresponds to a different CNN model used in the experiments:

- `maize-densenet121-final.ipynb`
- `maize-densenet169-final.ipynb`
- `maize-effecientnetb0-final.ipynb`
- `maize-effecientnetb1-final.ipynb`
- `maize-mobilenet-final.ipynb`
- `maize-mobilenetv2-final.ipynb`
- `maize-nasnetmobile-final.ipynb`

## Models Used

- DenseNet121
- DenseNet169
- EfficientNetB0
- EfficientNetB1
- MobileNet
- MobileNetV2
- NASNetMobile

These models were fine-tuned for maize disease classification using transfer learning techniques.

## Dataset

The dataset used consists of maize leaf images categorized by different disease types. Please note that the dataset itself is **not** included in this repository due to size and licensing restrictions. To run the notebooks, download the dataset from a trusted public source such as Kaggle and place it in the appropriate directory structure as specified in the notebooks.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn
- OpenCV (optional, for preprocessing/visualization)
