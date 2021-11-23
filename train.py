import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
import yaml
from addict import Dict

from build_loss import build_loss_func
from dataset import return_data
from evaluator import evaluator
from models import build_model

SEED = 14
torch.manual_seed(SEED)
torch.multiprocessing.set_sharing_strategy("file_system")


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def sweep(path):
    config = dict(yaml.safe_load(open(path)))
    sweep_id = wandb.sweep(config, project="ActionPurposeSegmentation")
    wandb.agent(sweep_id, main)


def main(sweep=False):
    config = get_arg()

    # config.save_folder = os.path.join(config.save_folder, config.model)
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    if config.wandb:
        name = config.model + "_" + config.head
        wandb.init(project="ActionPurposeSegmentation", config=config, name=name)
        config = wandb.config

    device = config.device
    # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("device:", device)
    # train_set, test_set = return_data.return_dataset(config)
    train_set, test_set = return_data.return_dataloader(config)
    model = build_model.build_model(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    weight = torch.ones(11).cuda()
    if True:
        weight[0] = 0
        weight[-1] = 0
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = build_loss_func(config, weight)

    best_eval = 0
    for epoch in range(1, 1 + config.epochs):
        print("\nepoch:", epoch)
        dataset_perm = np.random.permutation(range(len(train_set)))
        if config.repeat:
            for _ in range(config.repeat):
                dp = np.random.permutation(range(len(train_set)))
                dataset_perm = np.concatenate([dataset_perm, dp])

        t0 = time.time()
        train_loss = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset=train_set,
            config=config,
            device=device,
            dataset_perm=dataset_perm,
        )
        scheduler.step()
        print(f"\nlr: {scheduler.get_last_lr()}")
        t1 = time.time()
        print(f"\ntraining time :{round(t1 - t0)} sec")

        best_eval = test(
            model=model,
            dataset=test_set,
            config=config,
            device=device,
            best_eval=best_eval,
        )
        if config.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": train_loss,
                    "acc": best_eval,
                    "lr": scheduler.get_last_lr(),
                }
            )

    torch.save(model.state_dict(), "weight/model.pth")


def train(model, optimizer, criterion, dataset, config, device, dataset_perm):
    model.train()
    total_loss = 0
    counter = 0
    for data, label, mask, _ in dataset:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label, mask)
        print(f"\rloss: [{loss.item()}]", end="")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        counter += 1
    if True:
        a = output[0][mask[0] == 1]
        b = label[0, 20:]
        c = data[0][mask[0] == 1]
        print("")
        print("-----------------------")
        print(a[:10] * 120)
        print(b[:10] * 120)
        print(data[0, :10, 0] * 120)
    print(f"\rtotal_loss: [{total_loss / counter}]", end="")
    return total_loss / counter


def test(model, dataset, config, device, best_eval=0, th=0.6):
    model.eval()
    labels = []
    preds = []
    masks = []
    atts = []
    eval = evaluator()
    for data, label, mask, att in dataset:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)

        labels.append(label.detach().cpu().numpy()[:, :, 0])
        preds.append(output.detach().cpu().numpy())
        masks.append(mask.numpy())
        atts.append(att)

        # labels.extend(label.detach().cpu().numpy())
        # preds.extend(output.detach().cpu().numpy())

    # with open("test.pkl", "wb") as f:
    # pickle.dump(preds, f)
    eval.set_data(labels, preds, masks, atts)
    score = eval.evaluate()
    return max(score, best_eval)


if __name__ == "__main__":
    path = "config/config_tcn_sweep.yaml"
    main()
    # sweep(path)
