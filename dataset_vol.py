import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from PIL import Image


def default_loader(self, path):
    return Image.open(path).convert('RGB')

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
        self.root_dir = Path.db_dir(dataset)

        # dic ={'train': '1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54', \
        dic = {'train': '1 3 6', \
              'test': '4' }
                # 'val': '0 2 8 12 17 19 24 26 27 28 30 33 46 49 51', \
                # 'test': '4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47'}

        self.label_index = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3, 'l_winpoint': 4, \
        'l-pass': 5, 'l-spike': 6, 'l_set': 7}
        self.person_label_index = {'waiting': 0, 'setting': 1, 'digging': 2, 'falling': 3, 'spiking': 4, \
        'blocking': 5, 'jumping': 6, 'moving': 7, 'standing':8}
        video_index = dic[split].split(' ')
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 196#256#112#780
        self.resize_width = 128#384#112#1280
        # self.crop_size = 112

        self.fnames = []
        self.labels = []
        self.bbox = []
        self.transform = transforms
        self.preprocess_bbox(video_index)
        # self.loader = default_loader

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        labels = np.array(self.labels[index])
        buffer = self.load_frames(self.fnames[index], self.bbox[index])

        # if self.split == 'test':
        #     # Perform data augmentation
        #     buffer = self.randomflip(buffer)
        # buffer -= np.array([[[90.0, 98.0, 102.0]]])
        # buffer = self.normalize(buffer)
        # print("normalize", buffer.shape)
        buffer = self.transform(buffer)
        # buffer = self.to_tensor(buffer)
        # buffer_bbox = np.array(self.bbox[index])

        return buffer, torch.from_numpy(labels)
        # return torch.from_numpy(buffer), torch.from_numpy(labels)
        # return torch.from_numpy(buffer), torch.from_numpy(buffer_bbox), torch.from_numpy(labels)



    def preprocess_bbox(self, video_index):

        for video in video_index:
            # frames_list = os.listdir(os.path.join(self.root_dir, video))
            with open(os.path.join(self.root_dir, video, 'annotations.txt'),'r') as f:
                info = f.readlines()
                for item in info:
                    item_index = item.strip().split(' ')
                    # self.fnames.append(os.path.join(self.root_dir, video, item.split(' ')[0][:-4]))
                    # self.labels.append(label_index[item_index[1]])
                    for i in range(2, len(item_index), 5):
                        self.labels.append(self.person_label_index[item_index[i+4]])
                        self.fnames.append(os.path.join(self.root_dir, video, item_index[0][:-4], item_index[0]))
                        self.bbox.append(item_index[i:i+4])

        #sample label standing
        stand_index = np.where(np.array(self.labels)==8)
        # print(np.where(np.array(self.labels)!=8)[0].shape)
        use_index = np.concatenate((stand_index[0][::10], np.where(np.array(self.labels)!=8)[0]), axis=0)
        self.labels = np.array(self.labels)[use_index]
        self.fnames = np.array(self.fnames)[use_index]
        self.bbox = np.array(self.bbox)[use_index]

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
        return buffer.transpose((2, 0, 1))

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
                    item.split(' ')[0][:-4], 'person_detections.txt'))

        return frame_name, frame_label, frame_bbox

    def load_frames(self, frame_name, bbox_name):

        # frame = np.array(cv2.imread(frame_name)).astype(np.float64)
        frame = cv2.imread(frame_name)

        buffer = frame[int(bbox_name[1]):int(bbox_name[1])+int(bbox_name[3]), int(bbox_name[0]):int(bbox_name[0])+int(bbox_name[2])]

        person = cv2.resize(buffer, (self.resize_width, self.resize_height))
        # print("person", person.shape)
        buffer = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
        # buffer = np.array(cv2.resize(buffer, (self.resize_width, self.resize_height))).astype(np.float64)
        # cv2.imshow("person", person)
        # cv2.waitKey(0)

        return buffer

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
