class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            # root_dir = '/data/dataset/ucf101/UCF-101/'
            root_dir = '/data/dataset/ucf101/UCF-5/'
            # Save preprocess data into output_dir
            output_dir = '/data/dataset/VAR/UCF-5/'
            # output_dir = '/data/dataset/VAR/UCF-101/'
            bbox_output_dir = '/data/dataset/UCF-101-result/UCF-5-20/'

            return root_dir, output_dir, bbox_output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        elif database == 'something':
            root_dir = '/data/dataset/something-something-v2-lite/20bn-something-something-v2-frames'
            label_dir = '/data/dataset/something-something-v2-lite/label/something-something-v2-'
            bbox_output_dir = '/data/dataset/something-something-v2-lite/20bn-something-something-det'
            # root_dir = '/data/dataset/something-somthing-v2/20bn-something-something-v2-frames'
            # label_dir = '/data/dataset/something-somthing-v2/label/something-something-v2-'
            return root_dir, label_dir, bbox_output_dir
        elif database == 'volleyball':
            root_dir = '/mnt/data8tb/junwen/volleyball/videos'
            # bbox_dir = '/data/dataset/volleyball/'
            bbox_output_dir = '/mnt/data8tb/junwen/volleyball/volleyball-extra/'
            return root_dir, bbox_output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def group_dir(net):
        if net == 'vgg19':
            return '/mnt/data8tb/junwen/checkpoints/group_gcn/volleyball/vgg19_64_4096fixed_gcn2layer_lr0.01_pre71_mid5_lstm2/model_best.pth.tar'
 
    @staticmethod
    def model_dir(net):
        if net == 'vgg19':
            return '/home/junwen/opengit/player-classification-video/checkpoints/volleyball/vgg19_64_mid5_preImageNet_flip_drop/model_best.pth.tar'
        elif net == 'vgg19bn':
            return '/home/junwen/opengit/player-classification/checkpoints/volleyball/vgg19_bn_dropout/model_best.pth.tar' 
        elif net == 'alexnet':
            return '/home/junwen/opengit/player-classification/checkpoints/volleyball/alexnet/model_best.pth.tar'
