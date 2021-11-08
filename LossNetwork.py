import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
from data import TrainStation
from motsynth import MOTSynth, MOTSynthBlackBG
from log_utils import log_summary
from utils import save_ckpt, load_ckpt, print_scalor
from common import *
import parse
import pickle
from collections import namedtuple

# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        # self.layer_name_mapping = {
        #     '3': "relu1_2",
        #     '8': "relu2_2",
        #     '15': "relu3_3",
        #     '22': "relu4_3"
        # }
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2"
        }

        # self.layer_name_mapping = {
        #     '3': "relu1_2"        }
    def forward(self, x):
        # LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2"])
        # LossOutput = namedtuple("LossOutput", ["relu1_2"])
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)