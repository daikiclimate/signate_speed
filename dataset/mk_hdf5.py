from pathlib import Path
import numpy as np
import tqdm

import glob
import h5py
import os
import json
from tracking import add_track_annotations


src_path = "/home/ubuntu/data/signate_speed/data"
sequence_path = "/home/ubuntu/data/signate_speed/data/train_videos"
annotation_path = "/home/ubuntu/data/signate_speed/data/train_annotations"
test_sequence_path = "/home/ubuntu/data/signate_speed/data/test_videos"
test_annotation_path = "/home/ubuntu/data/signate_speed/data/test_annotations"

h5_path = "./data/hdf5_dataset_v2.h5"
h5_path = "./data/hdf5_dataset_v3.h5"
h5_path = "./data/hdf5_dataset_v5.h5"


def get_distance_image(raw_image_path, inf_DP, distance_limit=150):

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

    not_inf_index = img > inf_DP
    img[not_inf_index] = 560 / (img[not_inf_index] - inf_DP)
    img[img > distance_limit] = -1
    # img = img / distance_limit
    return img


def get_box_info(xywh, encode_sisa=True):
    x, y, w, h = xywh
    x_center = x + w // 2
    y_center = y + h // 2
    if encode_sisa:
        x_center = x_center // 4
        y_center = y_center // 4
    return int(x_center), int(y_center)


if __name__ == "__main__":
    speed_limit = 120
    distance_limit = 100
    with h5py.File(h5_path, "w") as h5:
        c = 0
        test_group = h5.create_group("test")
        for scene_index in tqdm.tqdm(sorted(os.listdir(test_sequence_path))):
            path = os.path.join(test_sequence_path, scene_index)

            subset_group = test_group.create_group(scene_index)
            anno_path = os.path.join(test_annotation_path, scene_index + ".json")
            with open(anno_path) as f:
                anno = json.load(f)

            video_path = f"data/test_videos/{scene_index}/Right.mp4"
            tracking_bboxes = add_track_annotations(video_path, anno)

            seq = anno["sequence"]
            inf_DP = seq[0]["inf_DP"]
            test_path = Path(f"data/test_videos/{scene_index}/disparity")
            test_raw_list = os.listdir(test_path)
            images = []
            for seq_index in range(len(seq)):
                test_raw_image_path = test_raw_list[seq_index]
                raw_image_path = test_path / test_raw_image_path
                img = get_distance_image(raw_image_path, inf_DP)
                img = img.reshape(105, -1)
                img = np.flip(img, 0)
                img = img[:, :250]
                images.append(img)
            subset_group.create_dataset("distance_image", data=images)

            feature = []
            target = []

            tgt_x = seq[0]["TgtXPos_LeftUp"]
            tgt_y = seq[0]["TgtYPos_LeftUp"]
            tgt_w = seq[0]["TgtWidth"]
            tgt_h = seq[0]["TgtHeight"]
            box = [tgt_x, tgt_y, tgt_w, tgt_h]
            x_center, y_center = get_box_info(box)
            tracking_box_center = [get_box_info(i) for i in tracking_bboxes]

            for s, img, tbox in zip(seq, images, tracking_box_center):
                t_x_center, t_y_center = tbox
                d_tracked = img[t_y_center, t_x_center]
                d = img[y_center, x_center]
                ownspeed = s["OwnSpeed"]
                # ownspeed /= speed_limit
                strdeg = s["StrDeg"] / 10
                inf_DP = s["inf_DP"]
                feature.append([ownspeed, strdeg, d, d_tracked])

            subset_group.create_dataset("feature", data=feature)

        train_group = h5.create_group("train")
        for scene_index in tqdm.tqdm(sorted(os.listdir(sequence_path))):
            # c += 1
            # if c == 10:
            #     pass
            path = os.path.join(sequence_path, scene_index)

            subset_group = train_group.create_group(scene_index)

            anno_path = os.path.join(annotation_path, scene_index + ".json")
            with open(anno_path) as f:
                anno = json.load(f)

            video_path = f"data/train_videos/{scene_index}/Right.mp4"
            tracking_bboxes = add_track_annotations(video_path, anno)

            att = anno["attributes"]
            att = att["評価値計算時の重み付加"]
            if att == "無":
                att = 1
            else:
                att = 3
            seq = anno["sequence"]
            inf_DP = seq[0]["inf_DP"]
            train_path = Path(f"data/train_videos/{scene_index}/disparity")
            train_raw_list = os.listdir(train_path)
            images = []
            for seq_index in range(len(seq)):
                train_raw_image_path = train_raw_list[seq_index]
                raw_image_path = train_path / train_raw_image_path
                img = get_distance_image(raw_image_path, inf_DP)
                img = img.reshape(105, -1)
                img = np.flip(img, 0)
                img = img[:, :250]
                images.append(img)
            subset_group.create_dataset("distance_image", data=images)

            feature = []
            target = []

            tgt_x = seq[0]["TgtXPos_LeftUp"]
            tgt_y = seq[0]["TgtYPos_LeftUp"]
            tgt_w = seq[0]["TgtWidth"]
            tgt_h = seq[0]["TgtHeight"]
            box = [tgt_x, tgt_y, tgt_w, tgt_h]
            x_center, y_center = get_box_info(box)
            tracking_box_center = [get_box_info(i) for i in tracking_bboxes]

            for s, img, tbox in zip(seq, images, tracking_box_center):
                d = img[y_center, x_center]
                t_x_center, t_y_center = tbox
                d_tracked = img[t_y_center, t_x_center]

                ownspeed = s["OwnSpeed"]
                # ownspeed /= speed_limit
                strdeg = s["StrDeg"] / 10
                inf_DP = s["inf_DP"]
                feature.append([ownspeed, strdeg, d, d_tracked])

                tgt_distance_ref = s["Distance_ref"]
                # tgt_distance_ref /= distance_limit
                tgt_speed_ref = s["TgtSpeed_ref"]
                # tgt_speed_ref /= speed_limit

                tgt_x_leftup = s["TgtXPos_LeftUp"]
                tgt_y_leftup = s["TgtYPos_LeftUp"]
                tgt_width = s["TgtWidth"]
                tgt_height = s["TgtHeight"]
                tgt = [tgt_speed_ref, tgt_distance_ref]

                target.append(tgt)
            subset_group.create_dataset("feature", data=feature)
            subset_group.create_dataset("target", data=target)
            subset_group.create_dataset("att", data=[att])
