# -*- coding:utf-8 -*-

# @Filename: Mixup
# @Project : ContestCD
# @date    : 2021-09-09 10:01
# @Author  : Linshan
import torch
import numpy as np


def mixup(inputs):
    batch_size = inputs[0].size(0)
    rand = torch.randperm(batch_size)

    lam = int(np.random.beta(0.2, 0.2) * inputs[0].size(2))
    new_inputs = []

    for input in inputs:
        rand_input = input[rand]
        new_input = torch.cat([input[:, :, 0:lam, :],
                               rand_input[:, :, lam:input.size(2), :]], dim=2)
        new_inputs.append(new_input)

    return new_inputs






