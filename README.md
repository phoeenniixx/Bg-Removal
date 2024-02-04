# Bg-Removal
### Input:
<img src ="https://github.com/phoeenniixx/Bg-Removal/assets/116151399/cd020722-1da6-4b3f-9658-7ea7ac7c4c07" height ="500" wodth="500">

### Output:
<img src ="https://github.com/phoeenniixx/Bg-Removal/assets/116151399/1795eca4-67df-4683-bc47-939bc2a4612b" height ="500" wodth="500">



## Project Description:

The project focuses on implementing a deep learning-based solution for background removal in images, particularly emphasizing person segmentation. The chosen architecture combines the strength of ResNet-50 for image feature extraction with a carefully designed encoder-decoder structure. Pre-trained on a large dataset like ImageNet, the ResNet-50 model forms the foundation for capturing intricate image features. The subsequent layers include Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Input, AveragePooling2D, and GlobalAveragePooling2D, orchestrating an effective encoder-decoder architecture.

## Key Components:

  1) ResNet-50:

  Pre-trained for robust feature extraction.
   
  2) Encoder-Decoder Architecture:

Combines Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Input, AveragePooling2D, and GlobalAveragePooling2D layers.


3) Conv2D Layers:

Extracts local patterns and features from the input image.


4) Batch Normalization:

Normalizes activations, aiding in training stability.


5) Activation Functions:

Introduces non-linearities for complex feature mapping.


6) Pooling Layers (MaxPool2D, AveragePooling2D):

Downsamples spatial dimensions, focusing on prominent features.


7) Global Average Pooling:

Reduces spatial dimensions to capture global context.


8) Conv2DTranspose Layers:

Upsamples and reconstructs spatial resolution in the decoder.
## Training and Optimization:

Trained on a person segmentation dataset.
Utilizes binary cross-entropy loss function.
Implements optimization techniques like gradient clipping, learning rate scheduling, and early stopping.
Initial learning rate set to 1e-4.
## Evaluation:

Assessed using metrics such as Intersection over Union (IoU), precision, recall, and F1 score on a validation set.
#### Final evaluation metrics:
    Loss: 0.0544
    Mean IoU: 0.3750
    Recall: 0.9593
    Precision: 0.9556
    Validation Loss: 0.1146
    Validation Mean IoU: 0.3784
    Validation Recall: 0.9279
    Validation Precision: 0.9215
    Learning Rate: 1.0000e-04

## Dataset: 

### Person Segmentation Dataset Description:

The person segmentation dataset, available at https://www.kaggle.com/datasets/nikhilroxtomar/person-segmentation/data, is a collection of images curated for the purpose of training and evaluating deep learning models for person segmentation tasks. This dataset is designed to facilitate the development of algorithms that can accurately identify and segment individuals within images.

### Key Characteristics:

1) Image Content:

The dataset comprises images featuring various scenes with individuals present.

2) Annotations:

Ground truth annotations in the form of segmented masks are provided for each image, outlining the regions corresponding to the persons within the scene.


3) Variety of Backgrounds:

The dataset includes a diverse set of backgrounds, challenging the model to effectively distinguish persons from varying environmental contexts.


4) Resolution and Quality:

Images in the dataset exhibit a range of resolutions and qualities, reflecting real-world scenarios and ensuring model robustness.


7) Dataset Size:

The dataset size, in terms of the number of images and corresponding annotations, is essential information for understanding the scale of the dataset and the potential for model generalization.

8) Training and Validation Split:

A clear separation between training and validation subsets allows for proper model training and evaluation. This ensures that the model's performance is assessed on unseen data.
