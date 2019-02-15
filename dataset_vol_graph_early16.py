import math
import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
# from network.model.graph_front.graphFront import _graphFront
from torchvision import transforms
from PIL import Image

class VolleyballDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='volleyball', split='train', transforms=transforms.ToTensor()):
        self.root_dir, self.bbox_output_dir = Path.db_dir(dataset)
        #dic = {'train': '1', \
        #      'val': '5', \
        #      'test': '4' }

        dic ={'train': '1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54', \
                'val': '0 2 8 12 17 19 24 26 27 28 30 33 46 49 51', \
                'test': '4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47'}

        label_index = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3, 'l_winpoint': 4, \
        'l-pass': 5, 'l-spike': 6, 'l_set': 7}
        video_index = dic[split].split(' ')
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 192#256#112#780
        self.resize_width = 112#384#112#1280

        self.transform2 = transforms

        self.fnames, self.labels, self.bboxes = self.make_dataset_sth(video_index, label_index)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        label = np.array(self.labels[index])
        pos_index = np.random.choice(np.where((self.labels==label))[0])
        neg_index = np.random.choice(np.where((self.labels!=label))[0])

        buffer, dist= self.load_frames(self.fnames[index], self.bboxes[index])
        pos_buffer, pos_dist= self.load_frames(self.fnames[pos_index], self.bboxes[pos_index])
        neg_buffer, neg_dist= self.load_frames(self.fnames[neg_index], self.bboxes[neg_index])

        #return torch.from_numpy(buffer), torch.from_numpy(label), torch.from_numpy(dist), \
        #torch.from_numpy(pos_buffer), torch.from_numpy(pos_dist), torch.from_numpy(neg_buffer), torch.from_numpy(neg_dist)
        return torch.from_numpy(buffer), torch.from_numpy(label), \
        torch.from_numpy(dist) #, torch.tensor(index)

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def make_dataset_sth(self, video_index, label_index):
        frame_name = []
        frame_label = []
        frame_bbox = []
        for video in video_index:
            with open(os.path.join(self.root_dir, video, 'annotations.txt'),'r') as f:
                info = f.readlines()
                for item in info:
                    #check
                    with open(os.path.join(self.bbox_output_dir, video, item.split(' ')[0][:-4], 'tracklets.txt'), 'r') as f:
                         det_lines = f.readlines()
                         if len(det_lines)<=8:
                            continue
                    item_index = item.split(' ')
                    frame_name.append(os.path.join(self.root_dir, video, \
                    item.split(' ')[0][:-4]))
                    frame_label.append(label_index[item_index[1]])
                    frame_bbox.append(os.path.join(self.bbox_output_dir, video, \
                    item.split(' ')[0][:-4], 'tracklets.txt'))

        return frame_name, frame_label, frame_bbox

    def load_frames(self, file_dir, bbox_dir):
        with open(bbox_dir, 'r') as f:
            det_lines = f.readlines()

        det_lines = [item.strip().split('\t') for item in det_lines]

        if len(det_lines) < 12:
            for i in range(12-len(det_lines)):
                det_lines.append(det_lines[-(i+1)])  #person number 12

        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)-16

        buffer = np.empty((frame_count, 12, 3, 109, 64), np.dtype('float32'))
        dist = np.zeros((frame_count, 3, 2, 12, 12), np.dtype('float64'))
        
        for idx in range(frame_count):
            i = idx
            frame_name = frames[i]
            frame = cv2.imread(frame_name)
            seq_x = np.zeros((12), dtype="float64")
            for j in range(len(det_lines)):
                buffer_bbox = [int(x) for x in det_lines[j][i+1].split(' ')]
                j_center_h = buffer_bbox[0]+buffer_bbox[1]/2
                seq_x[j] = j_center_h
            seq_index = np.argsort(seq_x)
            
            for l in np.arange(12):
                dist[idx,0,0,l,l] = 1/12
                #dist[idx,0,1,l,:l] = 1/12
                #dist[idx,0,1,l,l+1:] = 1/12
            dist[idx,0,1] = self.load_graph_12(seq_x)
            
            graph = dist[idx,0,1]
            if not np.isfinite(graph).all(): 
                print(seq_x)
                print(graph)
                print(frame_name)
            #for l in range(0,12,6):
            #    dist[idx,1,1,l:l+6,l:l+6] = 1/6
            for l in np.arange(12):
                dist[idx,1,0,l,l] = 1/6
                #dist[idx,1,1,l,l] = 0
            dist[idx,1,1] = self.load_graph(seq_x)
            graph = dist[idx,1,1]
            if not np.isfinite(graph).all(): 
                print(seq_x)
                print(graph)
                print(frame_name)

            for l in range(0,12,3):
                dist[idx,2,1,l:l+3,l:l+3] = 1/3

            for l in np.arange(12):
                dist[idx,2,0,l,l] = 1/3
                dist[idx,2,1,l,l] = 0

            #print(dist[0]) 
            for n in range(len(det_lines)):
                item = seq_index[n]
                buffer_bbox = [int(x) for x in det_lines[item][i+1].split(' ')]
                person = frame[buffer_bbox[1]:buffer_bbox[1]+buffer_bbox[3], \
                buffer_bbox[0]:buffer_bbox[0]+buffer_bbox[2]]
                #cv2.imshow('person', person)
                #cv2.waitKey(0)
                person = cv2.resize(person,(self.resize_width, self.resize_height))
                person = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
                person = self.transform2(person)
                buffer[idx][n][:] = person
            if len(det_lines)<12:
                buffer[idx][len(det_lines):] = buffer[idx][len(det_lines)-1]

        return buffer, dist

    def load_graph_12(self, seq_x):
        graph = np.zeros((12, 12), np.dtype('float64'))
        seq_x_sort = np.sort(seq_x)

        for i in range(len(seq_x_sort)):
            dis = 0
            for j1 in range(i):
                dis = dis + (seq_x_sort[i]-seq_x_sort[j1])
            for j2 in range(i+1,len(seq_x_sort)):
                dis = dis + (seq_x_sort[j2]-seq_x_sort[i])
            for j1 in range(i):
                graph[i,j1] = (dis-seq_x_sort[i]+seq_x_sort[j1])
            for j2 in range(i+1,len(seq_x_sort)):
                graph[i,j2] = (dis-seq_x_sort[j2]+seq_x_sort[i])

        if len(seq_x_sort)<12:
            for k in range(12-len(seq_x_sort),0,-1):
                graph[12-k,:] = graph[12-k-1,:]

        graph_sum = np.sum(graph, axis=1)
        graph = graph/graph_sum[:,np.newaxis]

        #print('graph', graph)
        #assert 1==0
        return graph*11/12

    def load_graph(self, seq_x):
        graph = np.zeros((12, 12), np.dtype('float64'))
        seq_x_sort = np.sort(seq_x)

        for i in range(math.ceil(len(seq_x_sort)/2)):
            dis = 0
            for j1 in range(i):
                dis = dis + (seq_x_sort[i]-seq_x_sort[j1])
            for j2 in range(i+1,6):
                dis = dis + (seq_x_sort[j2]-seq_x_sort[i])
            for j1 in range(i):
                graph[i,j1] = (dis-seq_x_sort[i]+seq_x_sort[j1])
            for j2 in range(i+1,6):
                graph[i,j2] = (dis-seq_x_sort[j2]+seq_x_sort[i])

            i0 = len(seq_x_sort)-1-i
            dis = 0
            for j1 in range(6, i0):
                dis = dis + (seq_x_sort[i0]-seq_x_sort[j1])
            for j2 in range(i0+1, 12):
                dis = dis + (seq_x_sort[j2] - seq_x_sort[i0])
            for j1 in range(6, i0):
                graph[12-i-1,j1] = (dis-seq_x_sort[i0]+seq_x_sort[j1])
            for j2 in range(i0+1, 12):
                graph[12-i-1,j2] = (dis-seq_x_sort[j2]+seq_x_sort[i0])

        if len(seq_x_sort)<12:
            for k in range((12-len(seq_x_sort))/2):
                graph[5-k,:] = graph[5-k-1,:]
                graph[6+k,:] = graph[6+k+1,:]

        #print('graph', graph)
        graph_sum = np.sum(graph, axis=1)
        #print('graph_sum', graph_sum)
        graph = graph/graph_sum[:,np.newaxis]

        return graph*5/6


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    root_dir = '/data/dataset/volleyball/videos/'
    train_data = VideoDataset(dataset='volleyball', split='test', clip_len=8, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
