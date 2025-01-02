# UAIC (Underwater Aquatic Image Classification)

## Overview
This repository contains the implementation of models for underwater aquatic image classification, primarily focusing on fish species. The project leverages supervised contrastive learning and includes both a lightweight custom model (based on depthwise separable convolutions) and a modified SimCLR framework.

## Features
- **Custom Lightweight Model:** Inspired by MobileNet, using depthwise separable convolutions for efficiency.
- **SimCLR Implementation:** Includes contrastive loss training for improved representation learning.
- **Data Augmentation:** Employs custom augmentations for robust training.
- **Pretrained Model Integration:** Option to use ResNet152 as a base encoder.
- **Performance Profiling:** Includes model profiling for operations like FLOPs.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Google Colab (optional)

Install dependencies:
```bash
pip install tensorflow matplotlib scikit-learn visualkeras
```

## Dataset
The dataset should be organized in subdirectories for each class. Update the path to the dataset in the `dataset_directory` variable in the script.

## Training

### Preprocessing
1. Resize images to 256x256.
2. Normalize pixel values.
3. Use data augmentation (random crops, flips, color jitter, etc.).

### Custom Lightweight Model
Run the script to train the lightweight model:
```python
simclr_model = create_custom_lightweight_model(input_shape, n_classes)
simclr_model.summary()
```

### SimCLR Framework
To train using the SimCLR model:
```python
simclr_model = create_simclr_model(input_shape, n_classes, model_path)
epoch_wise_loss, trained_model = train_simclr(optimizer, simclr_model, temperature=0.1)
```

### Fine-Tuning
If using a pretrained model (e.g., ResNet152):
- Set the `model_path` variable to the `.h5` file location.
- Fine-tune the model for your dataset.

## Visualization
Use `visualkeras` to visualize the model architecture:
```python
import visualkeras
visualkeras.layered_view(model, legend=True, draw_volume=False)
```

## Results
- The model’s performance is evaluated on train, validation, and test datasets.
- Training and validation metrics are plotted for accuracy and loss.

Example:
```python
plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
```

## Outputs
- Trained model saved as `.h5` files.
- Performance metrics and plots.

## Authors
Developed by 
- Poojyanth Reddy
- Srawan Meesala
- Tejas S


