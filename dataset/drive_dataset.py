import math
import h5py
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path
import json

from .cv_assigner import StKFoldAssigner


class DriveDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        config=None,
        transform=None,
        use_hdf5=False,
        cv_fold_num=0,
    ):
        src_path = Path("/home/ubuntu/data/signate_speed")
        self._annotation_path = src_path / "data/train_annotations"
        if mode == "test":
            self._annotation_path = src_path / "data/test_annotations"
        self._train_annotation_list = os.listdir(self._annotation_path)
        self._train_annotation_list = np.array(os.listdir(self._annotation_path))
        if not mode == "test":
            kfold_assigner = StKFoldAssigner("att", self._annotation_path)
            train_list, val_list = kfold_assigner(
                cv_fold_num, self._train_annotation_list
            )
            if mode == "train":
                self._train_annotation_list = self._train_annotation_list[train_list]
            elif mode == "val":
                self._train_annotation_list = self._train_annotation_list[val_list]

        self._config = config
        self._mode = mode
        self._distance_limit = 100
        self._data_type = "time"
        self._pad = 300
        self._speed_limit = 120
        self._use_hdf5 = use_hdf5

    def __getitem__(self, idx):
        if self._use_hdf5:
            if self._mode == "test":
                return self.get_test_hdf5(idx)
            else:
                return self.get_hdf5(idx)
        if self._data_type == "img":
            return self.get_image(idx)

        elif self._data_type == "time":
            return self.get_time_feature(idx)

    def get_hdf5(self, idx):
        seq_index = self._train_annotation_list[idx]
        seq_index = seq_index.split(".")[0]
        hdf_path = Path("/home/ubuntu/local/signate_speed/hdf5_dataset_v5.h5")
        f = h5py.File(hdf_path, "r")
        # f = h5py.File("data/hdf5_dataset.h5", "r")
        sample = f["train"][seq_index]
        img = sample["distance_image"]
        target = sample["target"]
        feature = sample["feature"]
        att = sample["att"][0]

        feature = torch.tensor(feature)
        feature[:, 0] /= 100
        feature[:, 1] /= 100
        feature[:, 2] /= 100

        self._feature_future = True
        self._feature_future = False

        if self._feature_future:
            time_diff = 10
            own_speed = feature[:, 0:1]
            _pad = torch.ones((time_diff, 1))
            _pad = _pad * own_speed[0]
            pad_speed = torch.cat([_pad, own_speed])
            r = torch.roll(pad_speed, time_diff, 0)[time_diff:]
            diff = own_speed - r
            feature = torch.cat([feature, diff], 1)

        target = torch.tensor(target)
        target[:, 0] /= 100
        target[:, 1] /= 100

        self._target_ratio = True
        if self._target_ratio:
            own_speed = feature[:, 0:1]
            tgt_speed = target[:, 0:1]
            tgt_ratio = tgt_speed / torch.clamp(own_speed, 0.01)
            target = torch.cat([target, tgt_ratio], 1)

        self._target_future_own = True
        # self._target_future_own = False
        if self._target_future_own:
            own_speed = feature[:, 0:1]
            own_future_speed = feature[:, 0:1]
            t = 10
            _pad = torch.ones((t, 1))
            _pad = _pad * torch.mean(own_future_speed)
            n_seq = len(own_future_speed)
            own_future_speed = torch.cat([own_future_speed, _pad])[t:]
            target = torch.cat([target, own_future_speed], 1)

        self._target_diff = True
        if self._target_diff:
            own_speed = feature[:, 0:1]
            tgt_speed = target[:, 0:1]
            tgt_diff = own_speed - tgt_speed
            target = torch.cat([target, tgt_diff], 1)

        mask = torch.zeros([self._pad])
        mask[: len(feature)] = 1
        mask[:20] = 0
        pad = torch.zeros([self._pad - len(feature), feature.shape[1]])
        feature = torch.cat([feature, pad])
        pad = torch.zeros(self._pad - len(target), target.shape[1])
        target = torch.cat([target, pad])

        return feature.float(), target.float(), mask.float(), att

    def get_test_hdf5(self, idx):
        seq_index = self._train_annotation_list[idx]
        seq_index = seq_index.split(".")[0]
        f = h5py.File("data/hdf5_dataset.h5", "r")
        sample = f["test"][seq_index]
        img = sample["distance_image"]
        feature = sample["feature"]

        feature = torch.tensor(feature)
        mask = torch.zeros([self._pad])
        mask[: len(feature)] = 1
        mask[:20] = 0
        pad = torch.zeros([self._pad - len(feature), feature.shape[1]])
        feature = torch.cat([feature, pad])
        return feature.float(), torch.tensor(0), mask.float(), seq_index

    def get_time_feature(self, idx):
        # idx = 0
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
        if self._mode == "test":
            att = 0
        else:
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
        # img = Image.open(raw_image_path)
        # img = np.array(img, dtype = np.float32)

        # not_inf_index = img > 0
        # inf_DP = seq[scene_index]["inf_DP"]
        not_inf_index = img > inf_DP
        img[not_inf_index] = 560 / (img[not_inf_index] - inf_DP)
        img[img > self._distance_limit] = 0
        img = img / self._distance_limit
        return img
        # dict_keys(['OwnSpeed', 'StrDeg', 'inf_DP', 'Distance_ref', 'TgtSpeed_ref', 'TgtXPos_LeftUp', 'TgtYPos_LeftUp', 'TgtWidth', 'TgtHeight'])

    def __len__(self):
        return len(self._train_annotation_list)


if __name__ == "__main__":
    d = DriveDataset()
