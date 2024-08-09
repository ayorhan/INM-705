# INM 705 Deep Learning for Image Analysis by Arda Yigithan Orhan

This project implements and evaluates two object detection models, Single Shot MultiBox Detector (SSD) and Faster R-CNN, using the COCO dataset. The goal is to identify everyday objects in real-time within a retail environment, a capability essential for autonomous retail systems like cashier-less stores.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training the Models](#training-the-models)
- [Running Inference](#running-inference)
- [Evaluation](#evaluation)
- [Known Issues](#known-issues)
- [Future Improvements](#future-improvements)

## Project Overview

This project focuses on developing an object detection system capable of identifying everyday objects in real-time within a retail environment. I employ two popular models, SSD and Faster R-CNN, both pre-trained on the COCO dataset, to detect and classify objects in images. The models were chosen for their balance between speed and accuracy, making them suitable for real-world retail applications.

## Dataset

The project uses the COCO (Common Objects in Context) dataset, which contains over 200,000 labeled images across 91 object categories. The dataset is accessed using the [Deeplake](https://docs.activeloop.ai/) library, which provides efficient data management and loading.

## Installation

To run this project, you need to install the required dependencies. Follow the steps below:

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/object-detection-retail.git
    cd object-detection-retail
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project is organized as follows:

```
project_root/
│
├── models/
│   ├── faster_rcnn.py       # Contains the Faster R-CNN model definition
│   ├── ssd.py               # Contains the SSD model definition
│
├── utils/
│   ├── dataset.py           # Dataset management and preprocessing functions
│
├── train.py                 # Script to train the models
├── inference.py             # Script to run inference using a trained model
├── requirements.txt         # Python package dependencies
├── setup.sh                 # Script to set up the environment and dependencies
├── README.md                # Project documentation (this file)
└── model.pth                # Saved model weights (generated after training)
```

## Training the Models

To train the models, run the `train.py` script. This script initializes the chosen model (either SSD or Faster R-CNN), loads the COCO dataset, and begins training.

```bash
python train.py
```

### Key Parameters

- `MODEL_TYPE`: Choose between `'ssd'` or `'faster_rcnn'`.
- `NUM_EPOCHS`: Set the number of training epochs.
- `BATCH_SIZE`: Set the batch size for training.
- `LEARNING_RATE`: Set the learning rate for the optimizer.
- `SAMPLE_SIZE`: Set the number of samples to use for training, useful for quick tests under limited computational resources.

## Running Inference

To run inference on a single image using a trained model, use the `inference.py` script. Specify the path to the image and the model type.

```bash
python inference.py --image path_to_image.jpg --model_type faster_rcnn
```

### Visualizing Results

The inference script will output the image with bounding boxes drawn around detected objects, displayed using Matplotlib.

## Evaluation

The evaluation during training is conducted using Intersection over Union (IoU) to measure the overlap between predicted bounding boxes and ground truth. The evaluation results are logged using [Wandb](https://wandb.ai/) for detailed analysis.


## Known Issues

- **No Valid IoU Scores**: During evaluation, the models fail to produce valid bounding boxes, leading to zero IoU scores. This could be due to incorrect data preprocessing or model initialization issues.
- **Limited Sample Size**: Due to computational constraints, a reduced sample size was used for training, which may limit the model's ability to generalize effectively.


## Future Improvements

- **Data Preprocessing**: Review and validate the data preprocessing steps to ensure correct bounding box transformations and label alignment.
- **Hyperparameter Tuning**: Experiment with different learning rates and batch sizes to improve model performance.
- **Sample Size**: Use a larger sample size for training to ensure the model encounters a diverse range of object instances and variations.
- **Model Fine-tuning**: Fine-tune the models more effectively for the COCO dataset to enhance detection accuracy.
