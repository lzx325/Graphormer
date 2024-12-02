import os
from pprint import pprint
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data import GraphDataModule, get_dataset
from model import Graphormer

if __name__=="__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Graphormer.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    print(args)
    pl.seed_everything(args.seed)

    dm = GraphDataModule.from_argparse_args(args)
    model = Graphormer(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        attention_dropout_rate=args.attention_dropout_rate,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate,
        weight_decay=args.weight_decay,
        ffn_dim=args.ffn_dim,
        dataset_name=dm.dataset_name,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        peak_lr=args.peak_lr,
        end_lr=args.end_lr,
        edge_type=args.edge_type,
        multi_hop_max_dist=args.multi_hop_max_dist,
        flag=args.flag,
        flag_m=args.flag_m,
        flag_step_size=args.flag_step_size,
    )
    dm.setup()
    loader=dm.train_dataloader()
    trainset=dm.dataset_train
    model(next(iter(loader)))

    # for i in np.random.choice(len(trainset),100,replace=False):
    #     print(trainset[i.item()].spatial_pos.max())
    