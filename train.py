import os


def main():
    cmd1 = "python cifar.py -a vgg19 | tee -a  /home/jc1088/Documents/opengit/volleyball/pytorch-classification/log.txt"
 #   cmd2 = "path/to/another_script | tee -a /path/to/logfile"

    os.system(cmd1)
 #   os.system(cmd2)


if __name__ == '__main__':
    main()
