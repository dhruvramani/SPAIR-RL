# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as tf


torch.manual_seed(0)

import cv2

def create(n_images=5000):
    trainings = []
    xtrainings = []
    testings = []
    vals = []

    w = 128
    h = 128
    fw = w
    fh = w

    _list = [
        "Frostbite-v0",
        #"Assault-v0",
        #"Asterix-v0",
        #"Carnival-v0",
        #"ElevatorAction-v0",
        #"Gopher-v0",
        #"Kangaroo-v0",
        #"SpaceInvaders-v0",
        #"Qbert-v0",
        #"MsPacman-v0",
    ]
    to_read = []
    for env in _list:
        for i in range(0, n_images):
            to_read.append("./atari_min/%s/%05d.jpg" % (env, i, ))

    import random
    random.shuffle(to_read)
    print(to_read[:10])
    for i, item in enumerate(to_read):
            img = cv2.imread(item)

            try:
                img = cv2.resize(img, (w, h))
            except:
                print(i)
                raise
            #st = 20
            #img = img[st: st + w, :]

            img = torch.from_numpy(np.expand_dims(img, axis=0).astype(np.float32)) / 255
            if i % 10 == 0:
                testings.append(img)
            elif i % 10 == 5:
                vals.append(img)
            else:
                trainings.append(img)

    train_imgs = torch.stack(trainings, dim=1).squeeze().permute(0, 3, 1, 2).view(-1, 3, fh, fw)
    test_imgs = torch.stack(testings, dim=1).squeeze().permute(0, 3, 1, 2).view(-1, 3, fh, fw)
    val_imgs = torch.stack(vals, dim=1).squeeze().permute(0, 3, 1, 2).view(-1, 3, fh, fw)
    print(train_imgs.shape)
    return train_imgs, test_imgs, val_imgs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Usage:')
    parser.add_argument('--nimages', default=5000, type=int,
                        help='number of images to create')
    args = parser.parse_args()
    train_set, test_set, val_set = create(args.nimages)
    torch.save(train_set, './train.pt')
    torch.save(val_set, './val.pt')
    torch.save(test_set, './test.pt')
