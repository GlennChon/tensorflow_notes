# TensorFlow - Course Outline

[Google CoLab](https://colab.research.google.com/drive/)

## Setup: 

_To run jupyter notebooks within WSL2_ :
[Setup instructions](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin)
[Enabling GPU acceleration on Ubuntu on WSL2](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview)

Switch over to directml Python environment
Enter command: `conda activate directml`

Uninstall conda's version of pillow if it is installed:
Enter command: `conda uninstall --force pillow`

Install pip version of pillow:
Enter command: `pip install pillow`

Restart Environment
Enter powershell command: `WSL --shutdown`

[1 - Fundamentals](01%20-%20Fundamentals.md)
* [.ipynb: 00_tensorflow_fundamentals](00_tensorflow_fundamentals.ipynb)
* [.ipynb: 01_neural_network_regression_in_tensorflow](01_neural_network_regression_in_tensorflow.ipynb)

[2 - Classification](02%20-%20Classification.md)
* [.ipynb: 02_neural_network_classification_in_tensorflow](02_neural_network_classification_in_tensorflow.ipynb)
* [Extra Resources](02_classification_challenge.md)

[3 - Computer Vision](03%20-%20Computer_Vision.md)
* [.ipynb: 03_convolutional_neural_networks_and_computer_vision](03_convolutional_neural_networks_and_computer_vision.ipynb)
* [.ipynb: data_modification](image_data_modification.ipynb)

[4 - Transfer Learning 1](04_transfer_learning_1.ipynb)

[5 - Transfer Learning 2](05_transfer_learning_2.ipynb)

[6 - Transfer Learning 3](06_transfer_learning_3.ipynb)

[7 - Food 101 Milestone Project](07+milestone_project_1.ipynb)