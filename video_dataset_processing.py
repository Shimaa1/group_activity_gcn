# This file defines a set of methods that are needed in creating a video dataset
import os
import sys
import fnmatch
import numpy as np
import torch.utils.data as data
import torch
from torch.autograd import Variable
import torch_extras

def GetProgressLabel(ratio):
    label = ratio * 10 - torch.ones(1)
    label = torch.Tensor.long(label)
    label = Variable(label).cuda()
    return label

# return a one-hot vector
def OneHot(label, num_class):
    size = (label.size(0), num_class)
    index = label.view(-1,1)
    onehot = torch_extras.one_hot(size, index.cpu()).cuda()
    return onehot

# combine one-hot vector with the data vector
def CombineSample(data, label, num_class):
    onehot = OneHot(label, num_class)
    if data.size(0) != label.size(0):
        onehot = onehot.repeat(data.size(0), 1)
    onehot = onehot.type(torch.cuda.FloatTensor)
    combined_data = torch.cat([data, onehot], 1)
    return combined_data

def CombineSample2(data, label, num_class):
    onehot = OneHot(label, num_class)
    onehot = onehot.unsqueeze(0)
    onehot = onehot.repeat(data.size(0), 1, 1)
    onehot = Variable(onehot.type(torch.cuda.FloatTensor))
    combined_data = torch.cat([data, onehot], 2)
    return combined_data

# return a partial video batch
def sample_data(data_batched, length_batched, ratio):
    #L*B*D
    
    length_batched = int(round(length_batched * ratio))
    data_batched = data_batched[:length_batched,:,:]

    #if ratio < 1.0:
    #    #length_batched = torch.round(length_batched.type(torch.FloatTensor) * ratio)
    #    #max_len = int(torch.max(length_batched))
    #    length_batched = round(length_batched * ratio)
    #    max_len = int(length_batched)

    #    for i in range(data_batched.size(1)):	
    #        index = int(length_batched[i])
    #        data_batched[index:,i,:] = 0

    #    data_batched = data_batched[0:max_len, :, :]

    return data_batched, length_batched

#
def sortData(length, data, label, cuda):

    length, indices = torch.sort(length, dim=0, descending=True)

    if cuda:
        data = data.cuda()
        indices = indices.cuda()
        label = label.cuda()

    data = data.index_select(1, indices)#[:, indices, :]
    label = label.index_select(0, indices)

    if not isSorted(length):
        print(length)
        raise RuntimeError("Batch not sorted!")

    return length, data, label

def modify_dataset_class_file(dataset_class_file):
    # change the label of the first action to 0
    file_path = os.path.dirname(dataset_class_file)
    new_dataset_class_file = os.path.splitext(os.path.basename(dataset_class_file))[0]
    new_dataset_class_file = new_dataset_class_file + "_MOD.txt"
    new_dataset_class_file = os.path.join(file_path, new_dataset_class_file)

    with open(dataset_class_file) as from_file:
        with open(new_dataset_class_file, "w") as to_file:
            for line in from_file:
                one_label, one_action = line.split()
                new_label = int(one_label) - 1
                line_to_write = str(new_label) + " " + one_action + "\n"
                to_file.write(line_to_write)

def isSorted(x):
    return all(x[i] >= x[i + 1] for i in xrange(len(x) - 1))

def mycollate(batch_list):
    # padding video data to make length of all the samples in a batch equal

    # find the maximum length in a batch
    max_length = 0
    batch_length = np.zeros(len(batch_list))
    for i in range(len(batch_list)):
        sample = batch_list[i][0]
        length = sample.shape[0]
        batch_length[i] = length
        if length > max_length:
            max_length = length

    # padding 0
    for i in range(len(batch_list)):
        sample = batch_list[i][0]
        length = sample.shape[0]
        pad_zero_len = max_length - length
        if pad_zero_len > 0:
            new_sample = np.lib.pad(sample, ((0, pad_zero_len), (0,0)), 'constant')
            batch_list[i][0] = new_sample

    return data.dataloader.default_collate(batch_list)

def read_class_file(dataset_class_file):
    # create action list and action to idx data based on text file dataset_class_file
    action_list = []
    action_to_idx = {}
    with open(dataset_class_file) as text_file:
        for line in text_file:
            one_label, one_action = line.split()
            action_to_idx[one_action] = one_label # order is not kept in this dictionary
            action_list.append(one_action)
    return action_list, action_to_idx

def gen_frm_file(data_path):
    # count the number of frames in a video, and write to a file
    dir_list = sorted(os.listdir(data_path))

    f = open('video_frm_ucf101.txt', 'w')

    for one_dir in dir_list:
        video_dir = data_path + '/' + one_dir
        num_frm = len(fnmatch.filter(os.listdir(video_dir), '*.npy'))

        string_to_write = one_dir + ' ' + str(num_frm) + '\n'
        f.write(string_to_write)

    f.close()

def read_a_video_from_feat_frms(data_path, action_name, video_name):
    # read a video in which each frame is represented as a feature in a npy file, and convert them into a single numpy array

    video_path_name = os.path.join(os.path.join(data_path, action_name), video_name)
    frm_files = sorted(fnmatch.filter(os.listdir(video_path_name), '*.npy'))
    num_frm = len(frm_files)
    frm_list = []

    for i in range(num_frm):
        # read in a frame feature
        frm_feat = np.load(os.path.join(video_path_name, frm_files[i]))
        # convert to a feature vector
        frm_feat = frm_feat.flatten()
        # add to a list
        frm_list.append(frm_feat)

    video_frm_array = np.array(frm_list)
    return video_frm_array

def read_a_video_from_video_npy(data_path, action_name, video_name, seq_max_len):
    # read a video which is a npy file
    video_path_name = os.path.join(os.path.join(data_path, action_name), video_name)
    video_array = np.load(video_path_name + ".npy")
    video_array = video_array[0:seq_max_len,:]
    return video_array

def save_a_video_to_npy(video_array, data_path, action_name, video_name):
    # data_path, action_name, video_name are for the destination
    video_path_name = os.path.join(os.path.join(data_path, action_name), video_name)
    np.save(video_path_name, video_array)

# create based on the setting file
def create_dataset(root, source): # root is the folder of containing the dataset, source is the setting file containing action names, labels, video names, #frames
    if not os.path.exists(source):
        print("Setting file %s for UCF-101 dataset does not exist." % (source))
        sys.exit()

    else:
        video_list = [] #
        with open(source) as split_file:
            data = split_file.readlines()
            for line in data:
                line_data = line.split()
                #video_path = os.path.join(root, line_data[0])
                action, video = line_data[0].split("/")

                #video_act = video.split("_")[1]

                #if action.lower() != video_act.lower():
                #    raise (RuntimeError("Setting file error"))

                num_frm = int(line_data[1])
                label = int(line_data[2])
                an_item = (action, label, video, num_frm)  # action name, action index (start from 0), video name, number of frames
                video_list.append(an_item)

    return video_list
