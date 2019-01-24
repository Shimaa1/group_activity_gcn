from __future__ import print_function
import torch
import os
import shutil
from torch.autograd import Variable


def get_pred_label(model_output):
    _, pred_label = torch.max(model_output.data, 1)
    return pred_label


def compute_acc(pred_label, true_label):
    total = pred_label.size(0)
    correct = (pred_label == true_label).sum()
    return float(correct) / total, correct

def save_checkpoint(state, is_best, filename, resume_path):
    if not os.path.exists(resume_path):
        os.mkdir(resume_path)
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)

def print_model(E_model, G1, G2, D_model):
    with open("model.txt", "w") as model_file:
        print("Encoder:\n", file=model_file)
        print(E_model, file=model_file)
        for param in E_model.parameters():
            print(type(param.data), param.size(), file=model_file)

        print("G1:\n", file=model_file)
        print(G1, file=model_file)
        for param in G1.parameters():
            print(type(param.data), param.size(), file=model_file)

        print("G2:\n", file=model_file)
        print(G2, file=model_file)
        for param in G2.parameters():
            print(type(param.data), param.size(), file=model_file)

        print("Discriminator:\n", file=model_file)
        print(D_model, file=model_file)
        for param in D_model.parameters():
            print(type(param.data), param.size(), file=model_file)

def norm_data(data):
     data_mean = torch.mean(data, dim=0)
     data_std = torch.std(data, dim=0)
     return (data-data_mean)/data_std

#def norm_data(data):
#    data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])
#    #data = 2 * data - 1
#    return data
