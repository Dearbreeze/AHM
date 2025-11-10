import numpy as np
import sys
import cv2


class State():
    def __init__(self, size, mask_size):
        self.image = np.zeros(size, dtype=np.float32)
        self.mask_size = mask_size

    def reset(self, x, mask):
        # self.image = np.clip(x + n, a_min=0., a_max=1.)  # add normal noise
        self.image = x
        self.mask = mask
        # size = self.image.shape
        # prev_state = np.zeros((size[0], 64, size[2], size[3]), dtype=np.float32)
        # self.tensor = np.concatenate([self.image, prev_state], axis=1)

    def set(self, x):
        self.image = x
        # self.tensor[:, :self.image.shape[1], :, :] = self.image


    def step(self, act,):
        self.mask = act.numpy()
        # print(self.image.shape)
        # print(self.mask_size)
        mask = self.mask.repeat(self.image.shape[2]/self.mask_size, axis=2).repeat(self.image.shape[3]/self.mask_size, axis=3)
        self.image = self.image * mask

        self.image = np.clip(self.image, a_min=0., a_max=1.)
        # self.tensor[:, :self.image.shape[1], :, :] = self.image
        # self.tensor[:, -64:, :, :] = inner_state
