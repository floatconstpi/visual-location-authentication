from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from glob import glob
import cv2
from pdb import set_trace as st
import numpy as np
from pycocotools.coco import COCO
import yaml
import numpy as np


def pinhole_distance_estimate(bb, scene=None):
    cam = cfg["scenes"][scene]

    K = np.array(cam["K"], dtype=np.float64)
    dist = np.array(cam["dist"], dtype=np.float64).reshape(-1, 1)
    actual_size = np.array(cam["size"], dtype=np.float64)


    fx = K[0, 0]

    # w_undistorted = undistorted_bbox_width(bb,K,dist)
    return float((fx * actual_size) / bb[2])


def undistorted_bbox_width(bb_xywh,K,dist):
    
    x, y, w, h = bb_xywh

    pts = np.array([
        [x,     y],
        [x + w, y],
        [x,     y + h],
        [x + w, y + h]
    ], dtype=np.float64).reshape(-1, 1, 2)

    pts_u = cv2.undistortPoints(pts, K, dist, P=K)  # shape (4,1,2)
    pts_u = pts_u.reshape(-1, 2)

    u_min, v_min = pts_u.min(axis=0)
    u_max, v_max = pts_u.max(axis=0)

    return float(u_max - u_min)


class DroneDataset(Dataset):
    def __init__(self, path, scene=None,split="train",predicted=False):
        self.path = path
        self.predicted = predicted
        self.metadata = pd.read_csv(os.path.join(path,"Distances.csv"))
        if scene is not None:
            self.images = list(self.metadata.loc[self.metadata['scene']==scene]["Photo"])
            
        else:
            self.images = list(self.metadata["Photo"])
        self.images = list(map(lambda x: os.path.join(path,"Drones data/images",x),self.images))
        
        if self.predicted:
            self.bb_info = pd.read_csv(os.path.join(path,"inference_bbox.csv"))
            self.images = [img for img in self.images if os.path.basename(img) in list(self.bb_info["filename"])] #to be removed and the two files reconciliated
        else:
            self.coco_ann = COCO(os.path.join(path,"annotations/instances.json"))
            self.file_to_imgid = {img["file_name"]: img["id"] for img in self.coco_ann.dataset["images"]}
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)

        if self.predicted:
            tmp = self.bb_info.loc[self.bb_info["filename"]==img_name]
            ann = [{"bbox":tmp.to_numpy()[0][1:-1]}]
            self.coco_ann = None

        else:
            img_id = self.file_to_imgid[img_name]
            ann_id = self.coco_ann.getAnnIds(imgIds=[img_id])
            ann = self.coco_ann.loadAnns(ann_id)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        distance = self.metadata.loc[self.metadata["Photo"]==img_name]["Distance"].item()
        scene = self.metadata.loc[self.metadata["Photo"]==img_name]["scene"].item()
        return image, distance, ann[0], self.coco_ann, scene, img_name