Vehicle Detection using YOLOv8

Overview

This project implements a vehicle detection system using the YOLOv8 (You Only Look Once) deep learning model. The model is trained to detect different types of vehicles such as cars, trucks, buses,bicycle,auto-rickshaw and motorcycles from images, videos, and real-time webcam input. The main goal of this project is to understand how object detection works and how YOLOv8 can be applied to real-world computer vision problems.

The project covers the complete workflow including dataset preparation, model training, and inference on different input sources.


Project Structure

The repository is organized in a simple and structured manner to make it easy to understand and use:
.
│
├── DATASET/ 
│   └── config.yaml
├── RESULT/
│
├── project_code.py/
│   ├── main.py
│   ├── predict_images.py
│   ├── predict_realtime.py
│   └── predict_video.py
│
├── .gitignore
├── Assignment1.py
├── README.md
└── requirements.txt

File Description

The main.py file is used for training the YOLOv8 model on the custom dataset. It loads a pretrained YOLOv8 model and fine-tunes it using the dataset defined in the configuration file.
The predict_images.py and predict_videos.py script performs vehicle detection on static images and videos respectively and saves the output with bounding boxes.
The predict_realtime_images.py script performs detection on live webcam input or video files.

The config.yaml file contains dataset paths and class names required by YOLO during training.
The requirements.txt file lists all Python libraries needed to run this project.
The .gitignore file prevents large files such as videos or unnecessary folders from being uploaded to GitHub.
ASSIGNMENT1.py containc solution for all the assignment1 question

Prerequisites

Ensure that Python version 3.8 or higher is installed on your system. A GPU is recommended for faster training but is not mandatory.


Training the Model

To train the YOLOv8 model on the dataset, run:

python src/main.py

The training configuration and dataset paths are defined in config/config.yaml.


Image/video Detection

To perform vehicle detection on images:

python src/predict_images.py

To perform vehicle detection on videos:

python src/predict_videos.py


Detected images with bounding boxes are saved in the outputs/predictions folder.


Real-Time / Video Detection

To perform detection using a webcam or video file:

python src/predict_realtime_images.py


Dataset Information

The dataset used in this project follows the YOLO annotation format. Each image has a corresponding label file containing bounding box coordinates and class information.

Classes: Car, Truck, Bus, Motorcycle,Auto-rickshaw,Bicycle
Dataset split: Training,Testing and Validation
Annotation format: YOLO (normalized coordinates)

The dataset is organized inside the dataset folder and is referenced using the dataset configuration file.


Outputs and Results

The result folder contains sample predictions, training curves, and evaluation metrics. Predicted images show detected vehicles with bounding boxes and confidence scores. Additional plots such as confusion matrices help in understanding the model’s performance.


Conclusion

This project helped in understanding the fundamentals of object detection using YOLOv8. It provided hands-on experience with training deep learning models, handling datasets, and performing inference on images and real-time data.
