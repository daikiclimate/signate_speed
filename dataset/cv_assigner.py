from sklearn.model_selection import StratifiedKFold
import numpy as np
import json


class StKFoldAssigner(object):
    def __init__(self, target, path):
        self._target = target
        self._path = path

    def __call__(self, fold_num, anno_path_list):
        targets = []
        for idx in range(len(anno_path_list)):
            seq_index = anno_path_list[idx]
            anno_path = self._path / seq_index
            seq_index = seq_index.split(".")[0]

            with open(anno_path) as f:
                anno = json.load(f)
            att = anno["attributes"]
            att = att["評価値計算時の重み付加"]
            if att == "無":
                att = 1
            else:
                att = 3
            targets.append(att)
        targets = np.array(targets)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=14)
        cv_value = np.zeros(len(targets))
        for i, (train_index, test_index) in enumerate(
            skf.split(np.zeros_like(targets), targets)
        ):
            cv_value[test_index] = i
        return cv_value != fold_num, cv_value == fold_num
