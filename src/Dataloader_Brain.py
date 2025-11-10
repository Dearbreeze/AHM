import os
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

from skimage import measure

from torch.autograd import Variable
# from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def PathList(path):


    # train_img_list = os.listdir(path + '/Train/img')
    # train_img_list.sort()
    # train_label_list = os.listdir(path + '/Train/anno')
    # train_label_list.sort()
    # test_img_list = os.listdir(path + '/Test/img')
    # test_img_list.sort()
    # test_label_list = os.listdir(path + '/Test/anno')
    # test_label_list.sort()

    train_img_list = os.listdir(path + '/Test/img')
    train_img_list.sort()
    train_label_list = os.listdir(path + '/Test/anno')
    train_label_list.sort()
    test_img_list = os.listdir(path + '/Train/img')
    test_img_list.sort()
    test_label_list = os.listdir(path + '/Train/anno')
    test_label_list.sort()


    # print(train_jpg_list[4],train_png_list[4])
    return train_img_list, train_label_list, test_img_list, test_label_list,


class MakeDataset(Dataset):
    def __init__(self, baseroot, imglist, annolist, train, img_size=None):
        self.baseroot = baseroot
        self.imglist = imglist
        self.annolist = annolist
        self.img_size = img_size
        self.train = train
        # self.transform1 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,), (0.5,))
        # ])
        # self.transform = transforms.Normalize((0.5,), (0.5,))
        # self.transform2 = transforms.Compose([
        #     transforms.RandomResizedCrop((128, 128)),
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # transforms.Compose([transforms.RandomResizedCrop((224, 224)),
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.ToTensor()])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        jpgpath = os.path.join(self.baseroot, 'img', self.imglist[index])
        pngpath = os.path.join(self.baseroot, 'anno', self.annolist[index])

        jpgimg_ori = cv2.imread(jpgpath, cv2.IMREAD_COLOR)
        # print(jpgimg_ori.shape,'000')
        jpgimg = cv2.cvtColor(jpgimg_ori, cv2.COLOR_BGR2RGB).transpose((2,0,1))
        # jpgimg = cv2.cvtColor(jpgimg, cv2.COLOR_RGB2GRAY)
        #
        pngimg = cv2.imread(pngpath, cv2.IMREAD_COLOR)
        # print(pngimg.shape)  (480, 640, 3)
        pngimg = cv2.cvtColor(pngimg, cv2.COLOR_BGR2RGB).transpose((2,0,1))
        # pngimg = cv2.cvtColor(pngimg, cv2.COLOR_RGB2GRAY)
        # jpgimg = jpgimg[58:426, 143:511]  # similar center crop to 368 x 368
        # pngimg = pngimg[58:426, 143:511]

        # if self.img_size is not None:
        #     jpgimg = cv2.resize(jpgimg, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
        #     pngimg = cv2.resize(pngimg, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
        # h, w = jpgimg.shape
        if self.train:
            if np.random.rand() > 0.5:
                jpgimg = np.fliplr(jpgimg)  # 左右方向上翻转
                pngimg = np.fliplr(pngimg)

        # if np.random.rand() > 1:
        #     angle = 10 * np.random.rand()
        #     if np.random.rand() > 0.5:
        #         angle *= -1
        #     M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)  # 中心旋转
        #     jpgimg = cv2.warpAffine(jpgimg, M, (w, h))  # cv2.getRotationMatrix2D(获得仿射变化矩阵) 2. cv2.warpAffine(进行仿射变化)
        #     pngimg = cv2.warpAffine(pngimg, M, (w, h))
        # print(jpgimg)
        # rand_range_h = h - self.crop_size
        # rand_range_w = w - self.crop_size
        # x_offset = np.random.randint(rand_range_w)
        # y_offset = np.random.randint(rand_range_h)
        # img = img[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size]
        # xs[i, 0, :, :] = (img / 255).astype(np.float32)
        # inputs = self.transform2(jpgimg)
        # targets = self.transform2(pngimg)
        # inputs = ((jpgimg - np.min(jpgimg)) / (np.max(jpgimg) - np.min(jpgimg))).astype(np.float32)
        inputs = (jpgimg / 255).astype(np.float32)
        targets = ((pngimg - np.min(pngimg)) / (np.max(pngimg) - np.min(pngimg))).astype(np.float32) # label value to 1
        # targets = pngimg.astype(np.float32)   # label value is 255
        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8))
        # ax0.imshow(inputs,cmap='gray')
        # ax1.imshow(targets,cmap='gray')
        # plt.show()
        # inputs = inputs[1, :, :]
        # inputs = inputs[np.newaxis,:,:]
        # targets = targets[np.newaxis,:,:]
        file_name = os.path.splitext(self.imglist[index])[0]
        # print(type(inputs))

        # image = np.asanyarray(inputs.transpose(1, 2, 0) * 255, dtype=np.uint8)
        # image = np.squeeze(image)
        # cv2.imwrite("test.png", image)

        return inputs, targets, file_name






