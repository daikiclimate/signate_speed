import numpy as np


class evaluator:
    def __init__(self, speed_limit=120):
        self._speed_limit = speed_limit

    def set_data(self, label, pred, mask, att):
        self.target = label
        self.pred = pred
        self.mask = mask
        self.att = att

    def evaluate(self):
        n_sample = len(self.pred)
        count = 0
        scores = []
        for i in range(n_sample):
            pred = self.pred[i]
            target = self.target[i]
            mask = self.mask[i]
            pred = pred[mask == 1]
            target = target[mask == 1]

            pred *= self._speed_limit
            target *= self._speed_limit

            att = self.att[i].numpy()

            x = np.abs((pred - target) / (0.07 * target + 3))
            x[x > 1] = 1
            s = np.mean(x)
            s *= att
            count += att
            scores.append(s)

        score = np.sum(scores) / count
        print("score:", score)
        return score
