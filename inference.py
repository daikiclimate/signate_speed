import sys

import numpy as np
import torch
import tqdm
import argparse
from addict import Dict
import yaml

from dataset import return_data
from models import build_model

import json


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    config = get_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = return_data.return_test_dataset(config)
    model = build_model.build_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load("weight/model.pth"))
    model.eval()

    labels = []
    preds = []

    with open("data/sample_submit.json") as f:
        sample_submit = json.load(f)
    for data, _, mask, sub_id in test_set:
        data = data.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(data) * 120
        mask[:20] = 1

        output = output[0, mask == 1].detach().cpu().numpy().tolist()
        sample_submit[sub_id] = output
    print("complete")
    with open("submit/sample_0.json", "w") as f:
        json.dump(sample_submit, f)


def count_labels(labels):
    num_labels = np.zeros(11, dtype=np.int)
    for l in labels:
        num_labels[l] += 1
    print(num_labels)


if __name__ == "__main__":
    main()
