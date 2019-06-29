# coding: utf-8

# In[ ]:


import os.path
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from PIL import Image
import PIL
from common import *


class CLEVR(Dataset):
    def __init__(self, root='./data', phase_train=True):

        if phase_train:
            key_word = 'train'
        else:
            key_word = 'val'

        json_fn = os.path.join(root, 'scenes', 'CLEVR_{}_scenes.json'.format(key_word))

        with open(json_fn) as json_file:
            json_data = json.load(json_file)

        self.num_obj = [len(s['objects']) for s in json_data['scenes']]

        self.image_path = os.path.join(root, 'images', '{}'.format(key_word))

        image_fn = [fn for fn in os.listdir(self.image_path) if
                    fn.startswith('CLEVR_') and fn.endswith('.png')]
        self.image_fn = sorted(image_fn, key=lambda s: int(s.split('.')[0][-6:]))

    def __getitem__(self, index):

        fn = self.image_fn[index]

        pil_img = Image.open(os.path.join(self.image_path, fn)).convert('RGB')
        # pil_img = pil_img.resize((320, 240), PIL.Image.BILINEAR)
        # pil_img = pil_img.crop((64, 29, 256, 221))

        pil_img = pil_img.crop((96, 16, 384, 304))

        image = np.array(pil_img.resize((img_w, img_h), PIL.Image.BILINEAR))

        image_t = torch.from_numpy(image / 255).permute(2, 0, 1).float()

        return image_t, torch.tensor(self.num_obj[index]).float()

    def __len__(self):
        return len(self.image_fn)

