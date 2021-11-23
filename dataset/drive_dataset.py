import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path
import json


class DriveDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        config=None,
        transform=None,
    ):
        src_path = Path("/home/ubuntu/data/signate_speed")
        self._annotation_path = src_path / "data/train_annotations"
        if mode == "test":
            self._annotation_path = src_path / "data/test_annotations"
        self._train_annotation_list = os.listdir(self._annotation_path)
        count = 0

        self._config = config
        self._mode = mode
        self._distance_limit = 100
        self._data_type = "time"
        self._pad = 300
        self._speed_limit = 120

    def __getitem__(self, idx):
        if self._data_type == "img":
            return self.get_image(idx)

        elif self._data_type == "time":
            return self.get_time_feature(idx)

    def get_time_feature(self, idx):
        idx = 0
        seq_index = self._train_annotation_list[idx]
        anno_path = self._annotation_path / seq_index
        seq_index = seq_index.split(".")[0]
        images = self.get_images(idx)

        with open(anno_path) as f:
            anno = json.load(f)
        if not self._mode == "test":
            att = anno["attributes"]
            att = att["評価値計算時の重み付加"]
            if att == "無":
                att = 1
            else:
                att = 3
        else:
            att = seq_index
        seq = anno["sequence"]
        feature = []
        target = []

        tgt_x = seq[0]["TgtXPos_LeftUp"]
        tgt_y = seq[0]["TgtYPos_LeftUp"]
        tgt_w = seq[0]["TgtWidth"]
        tgt_h = seq[0]["TgtHeight"]
        box = [tgt_x, tgt_y, tgt_w, tgt_h]
        x_center, y_center = self.get_box_info(box)
        for s, img in zip(seq, images):
            d = img[y_center, x_center]
            ownspeed = s["OwnSpeed"]
            ownspeed /= self._speed_limit
            strdeg = s["StrDeg"] / 10
            inf_DP = s["inf_DP"]
            feature.append([ownspeed, strdeg, d])
            if self._mode == "test":
                continue

            tgt_distance_ref = s["Distance_ref"]
            tgt_distance_ref /= self._distance_limit
            tgt_speed_ref = s["TgtSpeed_ref"]
            tgt_speed_ref /= self._speed_limit

            tgt_x_leftup = s["TgtXPos_LeftUp"]
            tgt_y_leftup = s["TgtYPos_LeftUp"]
            tgt_width = s["TgtWidth"]
            tgt_height = s["TgtHeight"]
            tgt = [tgt_speed_ref, tgt_distance_ref]

            target.append(tgt)
        feature = torch.tensor(feature)
        target = torch.tensor(target)
        mask = torch.zeros([self._pad])
        mask[: len(feature)] = 1
        mask[:20] = 0
        pad = torch.zeros([self._pad - len(feature), feature.shape[1]])
        feature = torch.cat([feature, pad])
        pad = torch.zeros(self._pad - len(target), target.shape[1])
        target = torch.cat([target, pad])
        return feature, target, mask, att

    def get_box_info(self, xywh, encode_sisa=True):
        x, y, w, h = xywh
        x_center = x + w // 2
        y_center = y + h // 2
        if encode_sisa:
            x_center = x_center // 4
            y_center = y_center // 4
        return int(x_center), int(y_center)

    def get_images(self, idx):
        seq_index = self._train_annotation_list[idx]
        anno_path = self._annotation_path / seq_index
        seq_index = seq_index.split(".")[0]

        train_path = Path(f"data/train_videos/{seq_index}/disparity")
        with open(anno_path) as f:
            anno = json.load(f)
        att = anno["attributes"]
        att = att["評価値計算時の重み付加"]
        if att == "無":
            att = 1
        else:
            att = 3
        seq = anno["sequence"]
        inf_DP = seq[0]["inf_DP"]
        train_raw_list = sorted(os.listdir(train_path))
        scene_index = random.randint(0, len(seq) - 1)
        images = []
        for scene_index in range(len(seq)):
            train_raw_image_path = train_raw_list[scene_index]
            raw_image_path = train_path / train_raw_image_path
            # inf_DP = seq[scene_index]["inf_DP"]
            img = self.get_distance_image(raw_image_path, inf_DP)
            img = img.reshape(105, -1)
            img = np.flip(img, 0)
            img = img[:, :250]
            if False:
                # if True:
                jpg_img = Image.fromarray(np.uint8(img * 255), "L")
                jpg_img.save(f"images/sample_difference_{scene_index}.jpg")
            images.append(img)
        return images

    def get_distance_image(self, raw_image_path, inf_DP):

        with open(raw_image_path, "rb") as f:
            disparity_image = f.read()
            # disparity_image = np.fromfile(f)
        img = []
        for d in range(0, len(disparity_image), 2):
            d_image = disparity_image[d]
            if d_image:
                d_image += disparity_image[d + 1] / 256
            img.append(d_image)
        img = np.array(img, dtype=np.float32)

        # train_path = Path(f"data/train_videos/{seq_index}/disparity_PNG")
        # train_raw_list = sorted(os.listdir(train_path))
        # train_raw_image_path = train_raw_list[scene_index]
        # raw_image_path = train_path / train_raw_image_path
        # print(raw_image_path)
        # img = Image.open(raw_image_path)
        # img = np.array(img, dtype = np.float32)

        # not_inf_index = img > 0
        # inf_DP = seq[scene_index]["inf_DP"]
        not_inf_index = img > inf_DP
        img[not_inf_index] = 560 / (img[not_inf_index] - inf_DP)
        img[img > self._distance_limit] = 0
        img = img / self._distance_limit
        return img
        # print(seq[scene_index].keys())
        # dict_keys(['OwnSpeed', 'StrDeg', 'inf_DP', 'Distance_ref', 'TgtSpeed_ref', 'TgtXPos_LeftUp', 'TgtYPos_LeftUp', 'TgtWidth', 'TgtHeight'])

    def __len__(self):
        return len(self._train_annotation_list)


if __name__ == "__main__":
    d = DriveDataset()