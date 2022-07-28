import torch
import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import cfg

class DatasetLoader(Dataset):
    def __init__(self, db, is_train, transform):
        
        self.db = db.data
        self.joint_num = db.joint_num
        self.root_idx = db.root_idx
        self.joints_have_depth = db.joints_have_depth
        
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False

    def __getitem__(self, index):
        
        joints_have_depth = self.joints_have_depth 
        data = copy.deepcopy(self.db[index])

        bbox = data['bbox']
        root_img = np.array(data['root_img'])
        root_vis = np.array(data['root_vis'])
        area = data['area']
        f = data['f']

        # 1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])
        img_height, img_width, img_channels = cvimg.shape
        
        # 2. get augmentation params
        if self.do_augment:
            rot, do_flip, color_scale = get_aug_config()
        else:
            rot, do_flip, color_scale = 0, False, [1.0, 1.0, 1.0]

        # 3. crop patch from img and perform data augmentation (flip, rot, color scale)
        img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, rot)
        for i in range(img_channels):
            img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

        # 4. generate patch joint, area_ratio, and ground truth
        # flip joints and apply Affine Transform on joints
        if do_flip:
            root_img[0] = img_width - root_img[0] - 1
        root_img[0:2] = trans_point2d(root_img[0:2], trans)
        root_vis *= (
                        (root_img[0] >= 0) & \
                        (root_img[0] < cfg.input_shape[1]) & \
                        (root_img[1] >= 0) & \
                        (root_img[1] < cfg.input_shape[0])
                        )
        
        # change coordinates to output space
        root_img[0] = root_img[0] / cfg.input_shape[1] * cfg.output_shape[1]
        root_img[1] = root_img[1] / cfg.input_shape[0] * cfg.output_shape[0]
        
        if self.is_train:
            img_patch = self.transform(img_patch)
            k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*f[0]*f[1]/(area))]).astype(np.float32)
            root_img = root_img.astype(np.float32)
            root_vis = root_vis.astype(np.float32)
            joints_have_depth = np.array([joints_have_depth]).astype(np.float32)

            return img_patch, k_value, root_img, root_vis, joints_have_depth
        else:
            img_patch = self.transform(img_patch)
            k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*f[0]*f[1]/(area))]).astype(np.float32)
          
            return img_patch, k_value

    def __len__(self):
        return len(self.db)