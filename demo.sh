python demo.py -e --gpu-id 'cuda:0' -a vgg19  -j 8 --train-batch 12 --lr 0.01 \
--resume "checkpoints/volleyball/vgg19_64_4096fixed_gcn2layer_lr0.01_pre71_mid5_lstm2_0/model_best.pth.tar" # > vgg19_64_4096fixed_gcn3step_lr0.01_pre71_mid5_lstm2.txt 
