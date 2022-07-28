import os.path as osp
import os
import sys
import numpy as np
from pycocotools.coco import COCO
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.utils import process_bbox
from config import cfg

class MSCOCO:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'MSCOCO', 'images')
        self.annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')
        self.human_bbox_dir = osp.join('..', 'data', 'MSCOCO', 'bbox_coco_output.json')
        self.joint_num = 19 # original: 17, but manually added 'Thorax', 'Pelvis'
        self.joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
        self.joints_have_depth = False

        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')
        self.lhip_idx = self.joints_name.index('L_Hip')
        self.rhip_idx = self.joints_name.index('R_Hip')
        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):

        if self.data_split == 'train':
            name = 'train2017'
        else:
            name = 'val2017'

        coco = COCO(osp.join(self.annot_path, 'person_keypoints_' + name + '.json'))
        data = []
        for ann_id in coco.anns.keys():
            ann = coco.anns[ann_id]
            img = coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']

            if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0):
                continue
            
            bbox = process_bbox(ann['bbox'], width, height)
            if bbox is None: continue
            area = bbox[2]*bbox[3]

            # joints and vis
            joint_img = np.array(ann['keypoints']).reshape(-1,3)
            # add Thorax
            thorax = (joint_img[self.lshoulder_idx, :] + joint_img[self.rshoulder_idx, :]) * 0.5
            thorax[2] = joint_img[self.lshoulder_idx,2] * joint_img[self.rshoulder_idx,2] # 1*1 일때만 visual하기 위해서
            thorax = thorax.reshape((1, 3))

            # add Pelvis
            pelvis = (joint_img[self.lhip_idx, :] + joint_img[self.rhip_idx, :]) * 0.5
            pelvis[2] = joint_img[self.lhip_idx,2] * joint_img[self.rhip_idx,2] # 1*1 일때만 visual하기 위해서
            pelvis = pelvis.reshape((1, 3))

            joint_img = np.concatenate((joint_img, thorax, pelvis), axis=0)

            joint_vis = (joint_img[:,2].copy().reshape(-1,1) > 0)
            joint_img[:,2] = 0

            root_img = joint_img[self.root_idx] # pelvis x,y, 0
            root_vis = joint_vis[self.root_idx] # pelvis True or False
 
            imgname = osp.join(name, img['file_name'])
            img_path = osp.join(self.img_dir, imgname)
            data.append({
                'img_path': img_path,
                'image_id': ann['image_id'],
                'bbox': bbox,
                'area': area,
                'root_img': root_img, # [org_img_x, org_img_y, 0]
                'root_vis': root_vis, # True of False
                'f': np.array([1500, 1500]), # dummy value
                'c': np.array([width/2, height/2]) # dummy value
            })

        return data

