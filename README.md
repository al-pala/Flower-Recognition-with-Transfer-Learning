# FlowerVision: AI Flower Classification with EfficientNet

A Computer Vision project for automatic Daisy and Dandelion classification using PyTorch, timm, transfer learning, and data augmentation.

## Overview

This project was developed for **GreenTech Solutions Ltd.**, an AgriTech company interested in implementing an automatic flower recognition system based on Artificial Intelligence and Computer Vision.

The objective is to develop a deep learning model capable of automatically classifying flower images into two categories:

- Daisy
- Dandelion

The proposed solution uses **PyTorch**, **timm**, and a pretrained **EfficientNet-B0** model. Transfer learning, data augmentation, partial fine-tuning, class-weighted loss, learning-rate scheduling, and early stopping are used to improve the model's performance and generalization.

The main objective of the project is to achieve the best possible **Macro F1-score** on the test set.

---

## Business Context

Automatic flower recognition can support several activities within an agricultural environment.

For GreenTech Solutions Ltd., a Computer Vision system could contribute to:

- Agricultural process optimization
- Automated plant monitoring
- Plant health assessment
- Data-driven agricultural decisions
- Reduction of repetitive manual activities
- Improved classification consistency
- More scalable agricultural monitoring

The project represents a prototype that could serve as a foundation for future Computer Vision applications in the AgriTech sector.

---

## Problem Statement

The provided dataset contains images belonging to two flower categories:

- **Daisy**
- **Dandelion**

The task is formulated as a supervised binary image classification problem.

The model receives an image as input and predicts one of the two possible flower classes.

The class mapping used in the project is:

| Flower | Label |
|---|---:|
| Daisy | 0 |
| Dandelion | 1 |

---

## Objective

The main objective is to develop an accurate and robust image classification model capable of distinguishing between Daisy and Dandelion images.

The primary evaluation metric is the **Macro F1-score** calculated on the test set.

Macro F1 was selected because the dataset is not perfectly balanced and the project should evaluate both classes fairly.

---

## Dataset

The dataset is divided into training, validation, and test sets.

### Dataset Distribution

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Daisy | 529 | 163 | 77 | 769 |
| Dandelion | 746 | 201 | 105 | 1,052 |
| **Total** | **1,275** | **364** | **182** | **1,821** |

The dataset contains more Dandelion images than Daisy images, resulting in a moderate class imbalance.

### Dataset Source

The dataset was provided as part of the project and is available at:

https://proai-datasets.s3.eu-west-3.amazonaws.com/progetto-finale-flowes.tar.gz

---

## Approach

The implemented solution follows a complete supervised Computer Vision pipeline:

1. Dataset loading
2. Dataset inspection
3. Class distribution analysis
4. Image preprocessing
5. Data augmentation
6. Transfer learning
7. Partial fine-tuning
8. Class-weighted loss
9. Model training
10. Validation
11. Learning-rate scheduling
12. Early stopping
13. Model checkpointing
14. Test evaluation
15. Confusion matrix analysis
16. Classification report
17. Single-image prediction

---

## Model Architecture

### EfficientNet-B0

The selected model is **EfficientNet-B0**, initialized with pretrained ImageNet weights through the `timm` library.

The final classifier is adapted to the two target classes.

The model is created using:

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=2
    )

EfficientNet-B0 was selected because the available dataset is relatively small.

Using a pretrained architecture allows the model to benefit from visual features learned on a much larger dataset instead of learning all representations from scratch.

---

## Transfer Learning

Transfer learning is used to reduce the risk of overfitting and improve training efficiency.

Initially, the pretrained model parameters are frozen and only the classifier is trainable.

The final part of the network is then fine-tuned by making the following components trainable:

- Final classifier
- Last two EfficientNet blocks
- Convolutional head

Most of the pretrained backbone remains frozen.

This strategy allows the model to preserve general visual representations while adapting the higher-level features to the specific flower classification task.

---

## Data Augmentation

The dataset is relatively small, so data augmentation is used to increase the variability of the training samples and improve generalization.

The training transformations include:

- Random resized crop
- Random horizontal flip
- Random rotation
- Color jitter
- Tensor conversion
- ImageNet normalization

The input images are transformed to a size of **224 × 224 pixels**.

The training transformation pipeline includes:

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

Validation and test images are not randomly augmented.

Instead, they are resized and center-cropped to ensure deterministic evaluation.

---

## Class Imbalance

The training set contains:

- 529 Daisy images
- 746 Dandelion images

Since the classes are not equally represented, class weights are calculated from the training distribution.

The weights are then used with the Cross Entropy Loss function:

    criterion = nn.CrossEntropyLoss(weight=class_weights)

This gives greater importance to the minority class during training and helps reduce the effect of the class imbalance.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Framework | PyTorch |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Backbone learning rate | 1e-5 |
| Classifier learning rate | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Maximum epochs | 25 |
| Early stopping patience | 3 |
| Loss function | Weighted Cross Entropy |

Two learning rates are used during training.

The classifier uses a higher learning rate because its parameters need to adapt to the new classification task.

The pretrained backbone uses a smaller learning rate to avoid excessively modifying the previously learned representations.

---

## Optimizer

The model is trained using **AdamW**.

The optimizer is configured with two parameter groups:

- Fine-tuned backbone parameters with a learning rate of `1e-5`
- Classifier parameters with a learning rate of `1e-4`

This allows the new classifier to learn more quickly while making smaller updates to the pretrained layers.

---

## Learning Rate Scheduler

A **CosineAnnealingLR** scheduler is used to progressively decrease the learning rate throughout training.

The configuration is:

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=25,
        eta_min=0.000001
    )

The scheduler helps reduce the learning rate as training progresses and can contribute to more stable convergence.

---

## Early Stopping

Early stopping is implemented to reduce overfitting.

The validation loss is monitored after every epoch.

Training stops if the validation loss does not improve for a predefined number of epochs.

The configuration used is:

- Patience: 3 epochs
- Minimum improvement: `0.0001`

This prevents unnecessary training once the model stops improving on the validation set.

---

## Model Checkpointing

The best model is saved according to the lowest validation loss.

The checkpoint is stored during training and can subsequently be loaded for the final test evaluation.

This ensures that the model used for evaluation corresponds to the best validation performance observed during training rather than necessarily the final training epoch.

---

## Evaluation Metrics

The model is evaluated on the independent test set using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Per-class Precision
- Per-class Recall
- Per-class F1-score
- Confusion Matrix

### Macro F1

Macro F1 is the main metric of the project.

For each class, Precision and Recall are used to calculate its F1-score.

The Macro F1-score is then obtained by averaging the F1-score of the two classes.

This gives Daisy and Dandelion equal importance, regardless of their different numbers of samples.

---

## Results

The final model is evaluated on the test set.

The main results obtained from the final training run should be reported below:

| Metric | Test Score |
|---|---:|
| Accuracy | **XX.XX%** |
| Macro Precision | **X.XXXX** |
| Macro Recall | **X.XXXX** |
| **Macro F1-score** | **X.XXXX** |

The exact values should be replaced with the metrics generated by the final execution of the notebook.

The project also generates a detailed classification report containing the performance of each individual class.

---

## Confusion Matrix

A confusion matrix is generated to analyze the model's predictions on the test set.

It shows the relationship between:

- True Daisy / Predicted Daisy
- True Daisy / Predicted Dandelion
- True Dandelion / Predicted Daisy
- True Dandelion / Predicted Dandelion

This allows the classification errors to be analyzed more clearly and helps determine whether the model performs consistently across both classes.

---

## Single Image Prediction

The project also includes an inference step using a randomly selected image from the test set.

For the selected image, the model produces:

- Predicted class
- Prediction probability
- Ground-truth class

The class probabilities are obtained by applying Softmax to the model's output logits.

This provides a simple example of how the trained model can be used for individual image classification.

---

## Training Curves

When training is enabled, the project records:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

These values are plotted to visualize the evolution of the model during training.

Training curves can be used to identify:

- Convergence
- Overfitting
- Underfitting
- Differences between training and validation performance

---

## Project Structure

    project/
    │
    ├── main/
    │   └── Python script
    │
    ├── notebook/
    │   └── Jupyter Notebook
    │
    └── README.md

The `main/` directory contains the Python implementation of the project.

The `notebook/` directory contains the complete experimental notebook, including dataset analysis, model development, training, evaluation, and visualization.

---

## Technologies

The project was developed using:

- Python
- PyTorch
- Torchvision
- timm
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- Google Colab

---

## Installation

Clone the repository:

    git clone https://github.com/andrealuigipala-del/Flower-Recognition-with-Transfer-Learning
    cd Flower-Recognition-with-Transfer-Learning

Create a virtual environment:

    python -m venv .venv

Activate the virtual environment on Linux/macOS:

    source .venv/bin/activate

On Windows:

    .venv\Scripts\activate

Install all project dependencies from `requirements.txt`:

    pip install -r requirements.txt

The `requirements.txt` file contains the Python packages required to run the project, including PyTorch, Torchvision, timm, NumPy, Scikit-learn, Matplotlib, Seaborn, and Pillow.

---

## Dataset Setup

Download the dataset:

    wget https://proai-datasets.s3.eu-west-3.amazonaws.com/progetto-finale-flowes.tar.gz

Extract the archive:

    tar -xzf progetto-finale-flowes.tar.gz

---

## Google Colab

The project was originally developed and tested using Google Colab.

The complete experimental workflow is available in the notebook.

Original Colab notebook:

https://colab.research.google.com/drive/1Scr0CE4UIQiAMpKJb_l3PXBx5vkX-xp6

---

## Key Design Decisions

### EfficientNet-B0

EfficientNet-B0 was selected because of the relatively small dataset size.

It provides a strong pretrained image representation while keeping the model relatively lightweight.

### Transfer Learning

Transfer learning allows the model to take advantage of representations learned from ImageNet.

This is particularly useful when the target dataset is too small to effectively train a deep neural network from scratch.

### Partial Fine-Tuning

Only the final part of the pretrained network is fine-tuned.

This provides domain adaptation while reducing the number of parameters that need to be updated.

### Data Augmentation

Data augmentation increases the variability of training images and helps the model become less dependent on specific image characteristics.

### Class-Weighted Loss

The dataset contains more Dandelion images than Daisy images.

Class-weighted Cross Entropy Loss is therefore used to reduce the impact of this imbalance.

### Macro F1

Macro F1 is used as the main evaluation metric because it gives equal importance to both flower classes.

---

## Limitations

The current implementation is a prototype and has several limitations.

### Dataset Size

The dataset contains only 1,821 images.

A larger dataset would likely provide more robust and generalizable results.

### Number of Classes

The current model only recognizes two flower categories:

- Daisy
- Dandelion

It is therefore not a general-purpose flower recognition system.

### Environmental Variability

Real-world images may contain conditions that are not fully represented in the dataset, including:

- Different lighting conditions
- Complex backgrounds
- Weather variations
- Camera quality differences
- Occlusions
- Different flower orientations
- Different flower development stages
- Different image resolutions

### Real-World Generalization

Additional data collected directly from agricultural environments would be valuable before deploying the model in a production environment.

---

## Future Improvements

Several improvements could be explored in future versions of the project.

### Model Comparison

Other architectures available through `timm` could be evaluated, such as:

- ResNet
- Other EfficientNet variants
- ConvNeXt
- MobileNet
- Vision Transformers

### Hyperparameter Optimization

A more systematic optimization process could be performed for:

- Learning rate
- Batch size
- Weight decay
- Number of unfrozen layers
- Scheduler parameters
- Data augmentation parameters

### Advanced Data Augmentation

Additional techniques could be investigated, including:

- Random Erasing
- MixUp
- CutMix

### Cross-Validation

K-fold cross-validation could provide a more robust estimate of the model's generalization performance, especially given the limited dataset size.

### Explainability

Techniques such as Grad-CAM could be implemented to visualize which parts of an image influence the model's prediction.

This would help determine whether the model is focusing primarily on the flower rather than on background features.

### Deployment

The trained model could eventually be integrated into:

- Agricultural monitoring systems
- Mobile applications
- Edge devices
- Greenhouse camera systems
- Automated crop-monitoring pipelines

---

## Business Impact

A production-ready version of this solution could provide several potential benefits to GreenTech Solutions Ltd.

### Increased Productivity

Automatic flower recognition could reduce the time required for manual identification.

### Improved Consistency

An automated model can apply the same classification process to large numbers of images.

### Scalability

The system could process increasing volumes of images without requiring a proportional increase in manual resources.

### Data-Driven Agriculture

Flower recognition could become one component of a broader agricultural Computer Vision system.

### Future Expansion

The same approach could potentially be extended to additional agricultural Computer Vision tasks, such as:

- Flower counting
- Plant species classification
- Disease detection
- Plant stress detection
- Crop monitoring
- Growth-stage classification

---

## Conclusion

This project demonstrates the development of a deep learning Computer Vision pipeline for automatic Daisy and Dandelion classification.

The implemented solution combines:

- PyTorch
- timm
- EfficientNet-B0
- ImageNet transfer learning
- Partial fine-tuning
- Data augmentation
- Class-weighted Cross Entropy Loss
- AdamW optimization
- Cosine learning-rate scheduling
- Early stopping
- Model checkpointing
- Macro F1 evaluation
- Confusion matrix analysis

The main challenge was the relatively small and moderately imbalanced dataset.

Transfer learning and data augmentation were therefore used to improve generalization, while partial fine-tuning allowed the pretrained model to adapt to the target classification problem.

Macro F1 was selected as the main metric because it evaluates the two classes equally.

The resulting solution provides a solid prototype for an AgriTech Computer Vision application and can serve as a starting point for future development involving larger datasets, additional flower or plant categories, explainability techniques, and real-world deployment.

---

## Author

**Andrea Luigi Pala**

Computer Vision & Deep Learning Project
