from BiNet import BiNet
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.metrics import confusion_matrix
import torch
import datetime
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from train import train
from test import test
from draw_result import draw_result
uncommon_pathogens_label = {
"C.afermentans":0,

"S.constellatus":1,


"S.saprophyticus":3,
"S.warneri":2,
"S.saliva":4,

"M.catarrhalis":5,

"S.anginosus":6
}

def set_parameter_requires_grad(model, feature_extracting):
    """
    是否保留梯度, 实现冻层
    :param model: 模型
    :param feature_extracting: 是否冻层
    :return: 无返回值
    """
    if feature_extracting:  # 如果冻层
        for param in model.parameters():  # 遍历每个权重参数
            param.requires_grad = False  # 保留梯度为False


def parameter_to_update(model, feature_extract=True):
    """
    获取需要更新的参数
    :param model: 模型
    :return: 需要更新的参数列表
    """
    print("Params to learn")
    param_array = model.parameters()

    if feature_extract:
        param_array = []
        #gg =model.named_parameters()
        for name, param, in model.named_parameters():
            if param.requires_grad == True:
                param_array.append(param)
                #print("\t", name)
    else:
        for name, param, in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    return param_array
args = {
    'batch_size': 32,
    'test_batch_size': 128,
    'epochs': 1,
    'lr': 0.0001,
    'momentum': 0.9,
    'seed': 1,
    'log_interval': 10
}
if __name__ == '__main__':
    train_dataset_file = r"\\fhy-2\分析文件\tcl\预处理所有数据uint64_line\NM_extrame_few\validation_map_NM_NOmask_delete.txt"
    test_dataset_file = r"\\fhy-2\分析文件\tcl\预处理所有数据uint64_line\NM_extrame_few\train_map_NM_NOmask_delete.txt"
    use_cuda = True if torch.cuda.is_available() else False
    # designating training device
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    model = BiNet().to(device)
    # load weights of common pathogens
    model.load_state_dict(torch.load("./model/common_pathogens.pth"))
    # set_parameter_requires_grad(model, feature_extract)  # 冻层

    # redesigning the classifier
    model.linear = torch.nn.Sequential(
        torch.nn.Linear(4608, 2304),
        torch.nn.ReLU(),
        torch.nn.Linear(2304, 1652),
        torch.nn.ReLU(),
        torch.nn.Linear(1652, len(uncommon_pathogens_label.keys())),
    )

    # 训练
    parameter_update = parameter_to_update(model)
    print(model)
    model = model.to(device)
    # starting training
    model = train(args, model, device, train_dataset_file, test_dataset_file)
    #saving model
    if not os.path.exists("./model"):
        os.mkdir("./model")
    torch.save(model.state_dict(), "./model/uncommon_pathogens.pth")
    #testing the trained model
    target, predict_y, g = test(args, model, device, test_dataset_file)
    draw_result(target, predict_y, uncommon_pathogens_label)
