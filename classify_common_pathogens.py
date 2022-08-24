from BiNet import BiNet
import torch
from train import train
from test import test
from draw_result import draw_result
import os
common_pathogens_label = {
"S.aureus" : 0,

"E.coli":1,

"P.aeruginosa":2,

"K.Pneumoniae":3,

"E.Faecium":4,

"A.baumannii":5,

"P.mirabilis":6,

"S.haemolyticus":7,

"S.maltophilia":8,

"E.faecium":9,

"K.aerogenes":10,

"S.marcescens":11,

"P.vulgaris":12,

"M.morganii":13,

"B.cepacia":14

}
args = {
    'batch_size': 128,
    'test_batch_size': 128,
    'epochs': 1,
    'lr': 0.0001,
    'momentum': 0.9,
    'seed': 1,
    'log_interval': 10
}
if __name__ == '__main__':
    train_dataset_dir_file = r"\\fhy-2\分析文件\tcl\预处理所有数据uint64_line\NM\train_map_NM_NOmask.txt"
    test_dataset_dir_file = r"\\fhy-2\分析文件\tcl\预处理所有数据uint64_line\NM\validation_map_NM_NOmask.txt"
    use_cuda = True if torch.cuda.is_available() else False
    # designating training device
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    model = BiNet().to(device)
    # starting training
    model = train(args, model, device, train_dataset_dir_file)
    # saving model
    if not os.path.exists("./model"):
        os.mkdir("./model")
    torch.save(model.state_dict(), "./model/common_pathogens.pth")
    # testing the trained model
    target, predict_y, g = test(args, model, device, test_dataset_dir_file)
    draw_result(target, predict_y, common_pathogens_label)