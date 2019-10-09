#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/9 21:14
# @Author:  Mecthew

from keras_lr_finder import LRFinder
from models import *
from utils.dataset import AutoSpeechDataset
from utils.data_process import ohe2cat
from utils.CONSTANT import *
import os

def find_lr():
    D = AutoSpeechDataset(os.path.join(r"C:\Users\90584\Desktop\Github\autospeech\sample_data\DEMO", 'DEMO.data'))
    D.read_dataset()
    metadata = D.get_metadata()
    x_train, y_train = D.get_train()
    my_model = CrnnModel()
    x_train = my_model.preprocess_data(x_train)
    y_train = ohe2cat(y_train)
    my_model.init_model(input_shape=x_train.shape[1:], num_classes=metadata[CLASS_NUM])

    lr_finder = LRFinder(my_model._model)
    lr_finder.find(x_train, y_train, start_lr=0.0001, end_lr=1, batch_size=64, epochs=5)


if __name__ == '__main__':
    find_lr()
