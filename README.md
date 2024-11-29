# PyMAD
Python Medical Anomaly Detection (PyMAD) using k-NNN and visibility detection 

This is an implementation of the k-NNN algorithm Anomaly detection in Medical Imaging as introduced in https://arxiv.org/abs/2305.17695 by Tal and Nizan.\
The visibility detection code can be found in https://github.com/CV-Reimplementation/Document-Enhancement-using-Visibility-Detection?tab=readme-ov-file

![visdetect](https://github.com/user-attachments/assets/72509863-e49b-4396-b73e-3382858e3f53)
*Visibility detection pipeline*

<img width="919" alt="Screenshot 2024-11-28 at 8 43 33 PM" src="https://github.com/user-attachments/assets/aef7547e-5d18-467d-9f9c-2e58ac8e6d93">

*k-NNN pipeline ([source](https://arxiv.org/abs/2305.17695))*

## Getting started
### Prepare your images
Store train and test images using the following structure:
```bash
dataset_mame/
├── train/
│   ├── img1
│   ├── img2
│   ├── ...
├── test/
│   ├── good/
│     ├── img1
│     ├── img2
│     ├── ...
│   ├── ungood/
│     ├── img1
│     ├── img2
│     ├── ...
```
### Running the algorithm
Main.py contains example code showing how to configure, train, and test the K-NNN model. 

## Results
To be updated soon
