#!/usr/bin/env python
import argparse
import importlib
import json
import os

from config import system_configs
from db.datasets import datasets
from nnet.py_factory import NetworkFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Local smoke train without multiprocessing")
    parser.add_argument("cfg_file", type=str, help="config file name without .json")
    parser.add_argument("--steps", type=int, default=2, help="number of train steps")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_path = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_path, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    dataset_name = system_configs.dataset
    train_split = system_configs.train_split
    print("cfg:", args.cfg_file)
    print("dataset:", dataset_name)
    print("train_split:", train_split)

    db = datasets[dataset_name](configs["db"], train_split)
    data_file = "sample.{}".format(db.data)
    sample_data = importlib.import_module(data_file).sample_data

    nnet = NetworkFactory(flag=True)
    nnet.cuda()
    nnet.set_lr(system_configs.learning_rate)
    nnet.train_mode()
    print("device:", nnet.device)

    k_ind = 0
    for iteration in range(1, args.steps + 1):
        batch, k_ind = sample_data(db, k_ind)
        set_loss, loss_dict = nnet.train(
            iteration=iteration,
            save=False,
            viz_split='train',
            **batch
        )
        loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value = loss_dict
        print(
            "step {} | total={:.4f} | object={:.4f} | class={:.4f} | loc={:.4f} | reg={:.4f} | range={:.4f}".format(
                iteration,
                float(loss_value),
                float(loss_dict_reduced.get('loss_object', 0.0)),
                float(loss_dict_reduced.get('loss_class', 0.0)),
                float(loss_dict_reduced.get('loss_loc', 0.0)),
                float(loss_dict_reduced.get('loss_reg', 0.0)),
                float(loss_dict_reduced.get('loss_range', 0.0)),
            )
        )
        del set_loss

    print("local smoke train completed")


if __name__ == "__main__":
    main()
