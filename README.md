# Zero Annotation Object Detection: Implementation on Fruits (In Progress)

## Overview
This project implements a fruit feature detection system using MobileNetV2 architecture. The system can identify and localize different types of fruits in images using advanced computer vision techniques including Grad-CAM (Gradient-weighted Class Activation Mapping) visualization and template matching.

## Features
- Fruit classification using MobileNetV2
- Feature visualization using Grad-CAM
- Template matching for object detection
- Bounding box generation
- Support for multiple fruit classes including apple, banana, cherry, chickoo, grapes, kiwi, mango, orange, and strawberry

## Requirements
- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- scikit-learn

## Dataset
- https://www.kaggle.com/datasets/shreyapmaher/fruits-dataset-images/data

## Requirements
```bash
# Versions
tensorflow>=1.15.2
keras==2.3.1
imutils==0.5.3
numpy==1.18.2
opencv-python==4.2.0.*
matplotlib==3.2.1
scipy==1.4.1
```

## Installation
```bash

# Install required packages
pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install Pillow
pip install scikit-learn
```


## Implementation Details

### 1. Model Architecture
The project uses MobileNetV2, a lightweight deep learning architecture suitable for mobile and embedded vision applications. The model has been trained to classify 9 different types of fruits.

### 2. Feature Detection Process
The feature detection pipeline consists of several key steps:

a) **Image Preprocessing and Classification**

- Images are resized to 224x224 pixels
- Pixel values are normalized using MobileNet's preprocessing function
- The image is then classified to generate the heatmap

b) **Feature Visualization**
- Implements Grad-CAM technique for visualizing important regions
- Generates heatmaps highlighting areas of interest
- Upscales heatmaps to match input image dimensions

c) **Template Matching**
- Extracts kernels from regions of high activation
- Performs template matching using OpenCV
- Uses adaptive thresholding for match detection

d) **Bounding Box Generation**
- Combines multiple detections into single bounding box
- Implements non-maximum suppression for overlap handling

## Usage 
All the below implementations are done in the `feature_heatmap_notebook.ipynd` notebook

1. Load the trained model:
```python
model = load_model('fruit_classifier.h5')
```
Or if any modifications needed in the model, modify the code in `Training_and_Testing.ipynb` notebook before training.

2. Prepare your image:
```python
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)
```

3. Generate feature heatmap:
```python
heatmap = make_gradcam_heatmap(image, model, "block_13_expand_BN")
```

4. Perform detection and visualization:
```python
# Code for template matching and bounding box drawing is provided in the notebook
```

## Future Improvements
- Implementation of K-means clustering for multiple object detection
- Enhanced template matching algorithms
- Real-time detection capabilities
- Improved bounding box accuracy

## Notes
- The current implementation works best with single fruit instances
- Model performance may vary based on image quality and lighting conditions
- Future updates will include handling multiple fruit instances
- Later a model with a robust algorith will be developed so that it will be working well on various datasets


## Developer
Pratyush Nakka
