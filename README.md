# Self-Supervised Medical Image Segmentation Using Deep Reinforced Adaptive Masking


### Prerequisites
We recommend Anaconda as the environment

* Linux Platform
* NVIDIA GPU + CUDA CuDNN
* Torch == 1.8.0
* torchvision == 0.9.0
* Python3.8.0
* numpy1.19.2
* opencv-python
* visdom

### Datasets
To download datasets:
- [Cardiac](https://www.kaggle.com/datasets/adarshsng/heart-mri-image-dataset-left-atrial-segmentation)
- [TCIA](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

### Training

To train the model, run this command:

```train
$ cd ./src/
$ python Train.py 
```
