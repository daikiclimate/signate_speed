import json
import os
from tqdm.notebook import tqdm

import cv2
import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt


def add_track_annotations(video_path, annotation):

    tracker = cv2.TrackerKCF.create()
    # with open(annotation_path, encoding="UTF-8") as f:
    #    annotation = json.load(f)
    sequences = annotation["sequence"]
    st_seq = sequences[0]
    tgt_x = st_seq["TgtXPos_LeftUp"]
    tgt_y = st_seq["TgtYPos_LeftUp"]
    tgt_w = st_seq["TgtWidth"]
    tgt_h = st_seq["TgtHeight"]
    xyxy = [tgt_x, tgt_y, tgt_w, tgt_h]

    # 動画ファイルを読込
    video = cv2.VideoCapture(video_path)

    # 動画情報
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))

    # 保存用
    # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # writer = cv2.VideoWriter(save_path, fmt, frame_rate, size)

    while True:
        ret, frame = video.read()
        if not ret:
            continue
        xyxy = np.array(xyxy).astype(int)
        tracker.init(frame, xyxy)
        break
    bboxes = [xyxy.tolist()]
    for frame_no in range(frame_count - 1):
        ret, image = video.read()
        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        debug_image = copy.deepcopy(image)

        ok, bbox = tracker.update(image)
        if ok:
            # 追跡後のバウンディングボックス描画
            # new_bbox = []
            # new_bbox.append(int(bbox[0]))
            # new_bbox.append(int(bbox[1]))
            # new_bbox.append(int(bbox[2]))
            # new_bbox.append(int(bbox[3]))
            # cv2.rectangle(debug_image,
            #             new_bbox,
            #             color_list[0],
            #             thickness=2)
            bboxes.append(bbox)
        else:
            bboxes.append(bboxes[-1])
    return bboxes
