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
        # self.root_dir, self.output_dir, self.bbox_output_dir = root_dir, output_dir, bbox_output_dir
        self.root_dir, self.bbox_output_dir = Path.db_dir(dataset)
        self.bbox_output = '/data/dataset/volleyball/person1/'
        # dic = {'train': '1 3 6', \
        #       'val': '5', \
        #       'test': '4' }

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

        # self.transform2 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #     ])
        self.fnames, self.labels, self.bboxes = self.make_dataset_sth(video_index, label_index)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        labels = np.array(self.labels[index])
        buffer, dist= self.load_frames(self.fnames[index], self.bboxes[index])
        # buffer = self.transform2(buffer)
        # buffer, buffer_bbox = self.crop(buffer, buffer_bbox, self.clip_len, self.crop_size)
        # adjacent_matrix = self.graph.build_graph(buffer_bbox[::2,:,:])

        # if self.split == 'test':
        #     # Perform data augmentation
        #     buffer = self.randomflip(buffer)
        # buffer = self.normalize(buffer)
        # buffer = self.to_tensor(buffer)

        # return torch.from_numpy(buffer), torch.from_numpy(labels), torch.from_numpy(buffer_bbox)
        # return torch.from_numpy(buffer), torch.from_numpy(buffer_bbox), \
        # torch.from_numpy(labels), adjacent_matrix
        return torch.from_numpy(buffer[::2,:,:,:,:]), torch.from_numpy(labels)
        # torch.from_numpy(dist[::2,:,:])

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
        frame_count = len(frames)

        buffer = np.empty((frame_count, 12, 3, 109, 64), np.dtype('float32'))
        dist = np.zeros((frame_count, 12, 12), np.dtype('float64'))

        for i, frame_name in enumerate(frames):
            frame = cv2.imread(frame_name)
            seq_x = np.zeros((12), dtype="float64")
            for j in range(len(det_lines)):
                buffer_bbox = [int(x) for x in det_lines[j][i+1].split(' ')]
                j_center_h = buffer_bbox[1]+buffer_bbox[3]/2
                seq_x[j] = j_center_h
            seq_index = np.argsort(seq_x)
            m = 0
            for item in seq_index:
                buffer_bbox = [int(x) for x in det_lines[item][i+1].split(' ')]
                person = frame[buffer_bbox[1]:buffer_bbox[1]+buffer_bbox[3], \
                buffer_bbox[0]:buffer_bbox[0]+buffer_bbox[2]]
                person = cv2.resize(person,(self.resize_width, self.resize_height))
                person = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
                person = self.transform2(person)
                buffer[i][m][:] = person
                m += 1

            # for j in range(len(det_lines)):
            #     buffer_bbox = [int(x) for x in det_lines[j][i+1].split(' ')]
            #     person = frame[buffer_bbox[1]:buffer_bbox[1]+buffer_bbox[3], \
            #     buffer_bbox[0]:buffer_bbox[0]+buffer_bbox[2]]
            #     j_center_h = buffer_bbox[1]+buffer_bbox[3]/2
            #     person = cv2.resize(person,(self.resize_width, self.resize_height))
            #     person = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
            #
            #     person = self.transform2(person)
            #     seq_x[j] = j_center_h
            #     buffer[i][j][:] = person
            # seq_index = np.argsort(seq_x)

            # for j in range(len(det_lines)):
            #     buffer_bbox = [int(x) for x in det_lines[j][i+1].split(' ')]
            #     person = frame[buffer_bbox[1]:buffer_bbox[1]+buffer_bbox[3], \
            #     buffer_bbox[0]:buffer_bbox[0]+buffer_bbox[2]]
            #     j_center_h = buffer_bbox[1]+buffer_bbox[3]/2
            #     j_center_w = buffer_bbox[0]+buffer_bbox[2]/2
            #     for k in range(j+1, len(det_lines)):
            #         buffer_bbox_k = [int(x) for x in det_lines[k][i+1].split(' ')]
            #         k_center_h = buffer_bbox_k[1]+buffer_bbox_k[3]/2
            #         k_center_w = buffer_bbox_k[0]+buffer_bbox_k[2]/2
            #         dist[i][j][k] = abs(j_center_h-k_center_h)+abs(j_center_w-k_center_w)
            #         dist[i][k][j] = dist[i][j][k]
            #
            #     person = cv2.resize(person,(self.resize_width, self.resize_height))
            #     person = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
            #
            #     person = self.transform2(person)
            #     buffer[i][j][:] = person

            dist_index = np.argsort(dist[i], axis=1)
            # for id in range(12):
            #     frame_show = frame
            #     pt_index = dist_index[id][:3]
            #     for jd in pt_index:
            #         buffer_bbox = [int(x) for x in det_lines[jd][id+1].split(" ")]
            #         frame_show = cv2.rectangle(frame_show, (buffer_bbox[0], buffer_bbox[1]), \
            #         (buffer_bbox[0]+buffer_bbox[2], buffer_bbox[1]+buffer_bbox[3]),(0,255,0),2)
            #     cv2.imshow("id", frame)
            #     cv2.waitKey(0)

            for l in np.arange(12):
                dist[i][l][dist_index[:, 3:][l]] = 1/12
                dist[i][l][dist_index[:, :3][l]] = 1/12

        return buffer, dist

    def crop(self, buffer, buffer_bbox, clip_len, crop_size):
        # randomly select time index for temporal jittering
        # time_index = np.random.randint(buffer.shape[0] - clip_len)
        #
        # # Randomly select start indices in order to crop the video
        # height_index = np.random.randint(buffer.shape[1] - crop_size)
        # width_index = np.random.randint(buffer.shape[2] - crop_size)

        time_index = 0
        # Randomly select start indices in order to crop the video
        height_index = 0
        width_index = 0
        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        buffer_bbox = buffer_bbox[time_index:time_index + clip_len, :]

        return buffer, buffer_bbox





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
