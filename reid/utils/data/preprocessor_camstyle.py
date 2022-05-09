from __future__ import absolute_import
import os.path as osp
import numpy as np
from PIL import Image
from ..serialization import read_json
import random

class PreprocessorCAM(object):
    def __init__(self, dataset, root=None, transform=None, num_cameras=None, mutual=True):
        super(PreprocessorCAM, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.cam_style_dir = osp.join(osp.join(self.root, ".."), "cam_style")
        self.fname2real_name = read_json(osp.join(osp.join(self.root, ".."), "fname2real_name.json"))
        self.num_cameras = num_cameras
        self.mutual = mutual

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        # if isinstance(indices, (tuple, list)):
        #     return [self._get_single_item(index) for index in indices]
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_mutual_item(self, index):
        fname, pid, _ = self.dataset[index]
        fpath = fname
        camid = int(fname.split("_")[1])
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        convert = np.random.rand() > 0.5
        if convert:
            while True:
                argue_camid = random.randint(1, self.num_cameras)
                if argue_camid != (camid + 1):
                    fpath_mix_up = osp.join(self.cam_style_dir, "{}_fake_{}to{}.jpg".format(self.fname2real_name[fname].split(".")[0], camid + 1, argue_camid))
                    img_cam = Image.open(fpath_mix_up).convert('RGB')
                    break
        else:
            img_cam = Image.open(fpath).convert('RGB')

        img = Image.open(fpath).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            img_cam = self.transform(img_cam)

        return img_cam, img, fname, pid, camid
    
    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid
